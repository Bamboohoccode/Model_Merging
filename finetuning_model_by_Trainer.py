"""Fine-tune a causal LM on stereotypical StereoSet sentences with Trainer.

The training defaults mirror the supplied script:
    batch size                  = 1 per device
    gradient accumulation      = 32
    maximum sequence length    = 512
    epochs                      = 30
    learning rate              = 3e-5
    weight decay               = 0.01
    warmup ratio               = 0.1

After training, the script can create an inverse/debiased model:
    theta_inverse = 2 * theta_base - theta_finetuned

Example on Kaggle (one process):
    python finetune_stereoset_trainer.py \
        --name_model ComCom/gpt2-small \
        --work_dir /kaggle/working

Two GPUs with FSDP (shard model/gradient/optimizer states):
    torchrun --nproc_per_node=2 finetune_stereoset_trainer.py \
        --name_model ComCom/gpt2-small \
        --work_dir /kaggle/working

Upload the inverse model (prefer setting HF_TOKEN in the environment):
    python finetune_stereoset_trainer.py --push_to_hub
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

import torch
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from huggingface_hub import HfApi
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)


# Hyperparameters from the supplied script.
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 32
MAX_LENGTH = 512
EPOCHS = 15
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
LEARNING_RATE = 3e-5

DEFAULT_HF_NAMESPACE = "trinhkhng"
DEFAULT_NAME_MODEL = "ComCom/gpt2-small"
DEFAULT_WORK_DIR = "/kaggle/working"

TRAIN_DTYPE = torch.float32
SAVE_DTYPE = torch.float16
STEREOTYPE_LABEL = 1


def build_stereotype_dataset() -> Dataset:
    """Return one text column containing only StereoSet stereotype examples."""
    intrasentence = load_dataset(
        "McGill-NLP/stereoset",
        "intrasentence",
        split="validation",
    )
    intrasentence = intrasentence.add_column(
        "type",
        ["intrasentence"] * len(intrasentence),
    )

    intersentence = load_dataset(
        "McGill-NLP/stereoset",
        "intersentence",
        split="validation",
    )
    intersentence = intersentence.add_column(
        "type",
        ["intersentence"] * len(intersentence),
    )

    merged = concatenate_datasets([intrasentence, intersentence])
    texts: list[str] = []

    for sample in merged:
        context = sample["context"].strip()
        sentences = sample["sentences"]["sentence"]
        labels = sample["sentences"]["gold_label"]

        for label, sentence in zip(labels, sentences, strict=True):
            if int(label) != STEREOTYPE_LABEL:
                continue

            sentence = sentence.strip()
            if sample["type"] == "intersentence" and context:
                text = f"{context} {sentence}"
            else:
                text = sentence

            if text:
                texts.append(text)

    if len(texts) < 2:
        raise RuntimeError(
            "Không lấy được đủ câu stereotype từ StereoSet; "
            "hãy kiểm tra phiên bản dataset và nhãn gold_label."
        )

    return Dataset.from_dict({"text": texts})


def prepare_dataset(
    tokenizer,
    max_length: int,
    eval_ratio: float,
    seed: int,
    preprocessing_num_workers: int | None,
) -> DatasetDict:
    """Split first, then tokenize both splits for causal language modeling."""
    if not 0.0 < eval_ratio < 1.0:
        raise ValueError("--eval_ratio phải nằm trong khoảng (0, 1).")

    raw_dataset = build_stereotype_dataset().train_test_split(
        test_size=eval_ratio,
        seed=seed,
        shuffle=True,
    )

    def tokenize(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
        )

    # remove_columns must be a list of source columns, not DatasetDict.column_names.
    tokenized = raw_dataset.map(
        tokenize,
        batched=True,
        remove_columns=["text"],
        num_proc=preprocessing_num_workers,
        desc="Tokenizing StereoSet",
    )
    return tokenized


def create_inverse_model(
    base_model_name: str,
    finetuned_dir: Path,
    inverse_dir: Path,
) -> None:
    """Create theta_inverse = 2*theta_base - theta_finetuned on CPU."""
    print("Đang tạo inverse model trên CPU...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    finetuned_model = AutoModelForCausalLM.from_pretrained(
        finetuned_dir,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )

    base_state = base_model.state_dict()
    finetuned_state = finetuned_model.state_dict()

    if base_state.keys() != finetuned_state.keys():
        missing = sorted(base_state.keys() - finetuned_state.keys())
        unexpected = sorted(finetuned_state.keys() - base_state.keys())
        raise ValueError(
            "Base model và fine-tuned model khác kiến trúc. "
            f"Missing={missing[:5]}, unexpected={unexpected[:5]}"
        )

    inverse_state: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for name, base_tensor in base_state.items():
            finetuned_tensor = finetuned_state[name]
            if torch.is_floating_point(base_tensor):
                inverse_state[name] = (
                    2.0 * base_tensor.float() - finetuned_tensor.float()
                )
            else:
                inverse_state[name] = base_tensor.clone()

    inverse_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    inverse_model.load_state_dict(inverse_state, strict=True)
    inverse_model.to(dtype=SAVE_DTYPE)
    inverse_model.config.use_cache = True
    inverse_model.save_pretrained(inverse_dir, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(inverse_dir)

    del base_state, finetuned_state, inverse_state
    del base_model, finetuned_model, inverse_model, tokenizer
    gc.collect()
    print(f"Đã lưu inverse model: {inverse_dir}")


def upload_model(
    model_dir: Path,
    repo_id: str,
    token: str | None,
) -> None:
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(model_dir),
        path_in_repo=".",
        commit_message=f"Upload {model_dir.name}",
    )
    print(f"Uploaded: https://huggingface.co/{repo_id}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a causal LM on StereoSet with Transformers Trainer."
    )
    parser.add_argument("--name_model", default=DEFAULT_NAME_MODEL)
    parser.add_argument("--work_dir", default=DEFAULT_WORK_DIR)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=GRADIENT_ACCUMULATION_STEPS,
    )
    parser.add_argument("--max_length", type=int, default=MAX_LENGTH)
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--warmup_ratio", type=float, default=WARMUP_RATIO)
    parser.add_argument(
        "--learning_rate_scheduler",
        choices=["linear", "cosine"],
        default="linear",
    )
    parser.add_argument("--eval_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging_steps", type=int, default=25)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--dataloader_num_workers", type=int, default=2)
    parser.add_argument("--preprocessing_num_workers", type=int, default=None)
    parser.add_argument(
        "--optim",
        default="adamw_torch",
        help="Trainer optimizer; adamw_torch works cleanly with FSDP.",
    )
    parser.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--skip_inverse", action="store_true")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hf-namespace", default=DEFAULT_HF_NAMESPACE)
    parser.add_argument("--repo_id", default=None)
    parser.add_argument("--hf_token", "--HF_TOKEN", dest="hf_token", default=None)

    # Accepted only so commands written for the old script do not fail.
    parser.add_argument(
        "--DEVICE",
        default=None,
        help="Deprecated: Trainer/torchrun places the model on devices automatically.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.DEVICE is not None:
        print(
            "Cảnh báo: --DEVICE không được dùng với Trainer. "
            "Dùng torchrun --nproc_per_node=2 nếu muốn train trên 2 GPU."
        )

    work_dir = Path(args.work_dir).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    model_short_name = args.name_model.rstrip("/").split("/")[-1]
    checkpoint_dir = work_dir / f"{model_short_name}_trainer_checkpoints"
    finetuned_dir = work_dir / f"{model_short_name}_finetuned"
    inverse_dir = work_dir / f"{model_short_name}_debias"

    tokenizer = AutoTokenizer.from_pretrained(args.name_model)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer không có cả pad_token lẫn eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dataset = prepare_dataset(
        tokenizer=tokenizer,
        max_length=args.max_length,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
        preprocessing_num_workers=args.preprocessing_num_workers,
    )
    print(
        f"StereoSet: train={len(dataset['train'])}, "
        f"eval={len(dataset['test'])}"
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.name_model,
        dtype=TRAIN_DTYPE,
        low_cpu_mem_usage=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = not args.gradient_checkpointing

    use_fp16 = bool(args.fp16 and torch.cuda.is_available())
    if args.fp16 and not torch.cuda.is_available():
        print("Không có CUDA; tự động tắt FP16.")

    training_args = TrainingArguments(
    output_dir=str(checkpoint_dir),
    num_train_epochs=args.epochs,

    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,

    gradient_accumulation_steps=args.gradient_accumulation_steps,
    accelerator_config={
        "gradient_accumulation_kwargs": {
            "sync_each_batch": True,
        }
    },

    learning_rate=args.learning_rate,
    weight_decay=args.weight_decay,

    # Transformers 5.0: float < 1 vẫn được hiểu là ratio.
    # args.warmup_ratio = 0.1 -> warmup 10%
    warmup_steps=args.warmup_ratio,

    lr_scheduler_type=args.learning_rate_scheduler,

    # QUAN TRỌNG: giảm mạnh optimizer states trên GPU
    optim="paged_adamw_8bit",

    max_grad_norm=1.0,
    fp16=use_fp16,

    gradient_checkpointing=False,

    fsdp=True,
    fsdp_config={
        "version": 1,
        "activation_checkpointing": True,

        "auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
        "transformer_layer_cls_to_wrap": ["LlamaDecoderLayer"],

        # Giảm peak memory lúc backward
        "backward_prefetch": "BACKWARD_POST",
        "limit_all_gathers": True,

        "state_dict_type": "FULL_STATE_DICT",
    },
    save_only_model=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=args.logging_steps,
    save_total_limit=args.save_total_limit,

    load_best_model_at_end=False,
    prediction_loss_only=True,

    report_to="none",

    dataloader_num_workers=args.dataloader_num_workers,
    dataloader_pin_memory=True,

    seed=args.seed,
    data_seed=args.seed,
)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8 if use_fp16 else None,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
    )
    print("FSDP enabled:", trainer.is_fsdp_enabled)
    print("Distributed:", trainer.accelerator.distributed_type)
    train_result = trainer.train(
        resume_from_checkpoint=args.resume_from_checkpoint
    )
    trainer.model.config.use_cache = True
    trainer.save_model(str(finetuned_dir))
    trainer.save_state()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)

    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(finetuned_dir)
        summary = {
            "base_model": args.name_model,
            "train_examples": len(dataset["train"]),
            "eval_examples": len(dataset["test"]),
            "effective_batch_size_per_process": (
                args.batch_size * args.gradient_accumulation_steps
            ),
            "hyperparameters": {
                "epochs": args.epochs,
                "per_device_batch_size": args.batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "max_length": args.max_length,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "warmup_ratio": args.warmup_ratio,
                "lr_scheduler_type": args.learning_rate_scheduler,
                "optimizer": args.optim,
                "fp16": use_fp16,
            },
            "train_metrics": train_result.metrics,
            "eval_metrics": eval_metrics,
        }
        with (finetuned_dir / "training_summary.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(summary, file, ensure_ascii=False, indent=2)

    trainer.accelerator.wait_for_everyone()
    is_main_process = trainer.is_world_process_zero()

    del trainer, model, dataset, data_collator
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Only rank 0 creates or uploads the final inverse model.
    if not is_main_process:
        return

    if not args.skip_inverse:
        create_inverse_model(
            base_model_name=args.name_model,
            finetuned_dir=finetuned_dir,
            inverse_dir=inverse_dir,
        )

    if args.push_to_hub:
        upload_dir = finetuned_dir if args.skip_inverse else inverse_dir
        repo_id = args.repo_id or (
            f"{args.hf_namespace}/debias_{model_short_name}"
            if not args.skip_inverse
            else f"{args.hf_namespace}/{model_short_name}_finetuned"
        )
        token = args.hf_token or os.environ.get("HF_TOKEN")
        upload_model(upload_dir, repo_id, token)


if __name__ == "__main__":
    main()
