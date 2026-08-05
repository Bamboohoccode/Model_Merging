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

Two GPUs with DistributedDataParallel:
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
import math
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
from transformers.integrations.deepspeed import unset_hf_deepspeed_config


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

TRAIN_DTYPE = torch.float16
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
        default="paged_adamw_8bit",
        help="Trainer optimizer; use adamw_torch if bitsandbytes is unavailable.",
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

    use_fp16 = bool(args.fp16 and torch.cuda.is_available())
    if args.fp16 and not torch.cuda.is_available():
        print("Không có CUDA; tự động tắt FP16.")

    world_size = max(1, int(os.environ.get("WORLD_SIZE", "1")))
    samples_per_process = math.ceil(len(dataset["train"]) / world_size)
    updates_per_epoch = max(
        1,
        math.ceil(
            samples_per_process / args.gradient_accumulation_steps
        ),
    )
    warmup_steps = int(
        args.warmup_ratio * updates_per_epoch * args.epochs
    )

    deepspeed_config = {
        "fp16": {"enabled": "auto"},
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": "auto",
                "betas": "auto",
                "eps": "auto",
                "weight_decay": "auto",
            },
        },
        "zero_optimization": {
    "stage": 3,

    "offload_optimizer": {
        "device": "nvme",
        "nvme_path": nvme_path,
        "pin_memory": False,
        "buffer_count": 4,
        "fast_init": False,
    },

    "offload_param": {
        "device": "nvme",
        "nvme_path": nvme_path,
        "pin_memory": False,
        "buffer_count": 5,
        "buffer_size": 100_000_000,
        "max_in_cpu": 100_000_000,
    },

    "overlap_comm": True,
    "contiguous_gradients": True,
    "sub_group_size": 100_000_000,
    "reduce_bucket_size": 50_000_000,
    "stage3_prefetch_bucket_size": 50_000_000,
    "stage3_param_persistence_threshold": 100_000,
    "stage3_max_live_parameters": 100_000_000,
    "stage3_max_reuse_distance": 100_000_000,
    "stage3_gather_16bit_weights_on_model_save": True,
},
    "aio": {
        "block_size": 262144,
        "queue_depth": 32,
        "thread_count": 1,
        "single_submit": False,
        "overlap_events": True,
    },
        "gradient_clipping": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": "auto",
        "train_batch_size": "auto",
    }
    nvme_path = str(work_dir / "deepspeed_nvme")
    Path(nvme_path).mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=warmup_steps,
        lr_scheduler_type=args.learning_rate_scheduler,
        max_grad_norm=1.0,
        fp16=use_fp16,
        bf16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        deepspeed=deepspeed_config,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        prediction_loss_only=True,
        report_to="none",
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=False,
        seed=args.seed,
        data_seed=args.seed,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.name_model,
        dtype=TRAIN_DTYPE if use_fp16 else torch.float32,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

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
            "effective_batch_size_per_process": args.gradient_accumulation_steps,
            "hyperparameters": {
                "epochs": args.epochs,
                "per_device_batch_size": 1,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "max_length": args.max_length,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "warmup_ratio": args.warmup_ratio,
                "lr_scheduler_type": args.learning_rate_scheduler,
                "optimizer": "DeepSpeedCPUAdam",
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

    del trainer, training_args, model, dataset, data_collator
    unset_hf_deepspeed_config()
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
