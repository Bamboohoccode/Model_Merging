import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import DatasetDict, load_dataset,concatenate_datasets
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm
from pathlib import Path
import argparse,os
import shutil
import gc
import math
from bitsandbytes.optim import PagedAdamW8bit
## CONFIG ==========================================================================================
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 32
max_length = 512
EPOCHS = 30
weight_decay = 0.01
warmup_ratio = 0.1
DEFAULT_lr = 3e-5
DEFAULT_HF_NAMESPACE = "trinhkhng"
DEFAULT_NAME_MODEL = "ComCom/gpt2-small"
DEFAULT_WORK_DIR = "/kaggle/working/"
TRAIN_DTYPE = torch.float32
SAVE_DTYPE = torch.float16
class StereoSet_DataSet(Dataset):
    def __init__(self, inputs, tokenizer, max_length=512):
        self.inputs = inputs
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __getitem__(self, idx):
        return self.tokenizer(
            self.inputs[idx],
            truncation=True,
            max_length=self.max_length,
        )

    def __len__(self):
        return len(self.inputs)

def Return_DataLoader(tokenizer,batch_size = 8):

    stereoset = DatasetDict({
        "intrasentence": load_dataset(
            "McGill-NLP/stereoset",
            "intrasentence",
            split="validation",
        ),
        "intersentence": load_dataset(
            "McGill-NLP/stereoset",
            "intersentence",
            split="validation",
        ),
    })
    intra = stereoset["intrasentence"].add_column(
        "type",
        ["intrasentence"] * len(stereoset["intrasentence"])
    )

    inter = stereoset["intersentence"].add_column(
        "type",
        ["intersentence"] * len(stereoset["intersentence"])
    )
    merged_dataset = concatenate_datasets([intra, inter])
    preprocessed_inputs = []
    for sample in merged_dataset:
        context = sample['context']
        label = sample["sentences"]['gold_label']
        sentences = sample['sentences']['sentence']
        for lb,sen in zip(label,sentences):
            if(sample['type'] == 'intrasentence'):
                input_text = sen
            else:
                input_text = (
                    context.strip()
                    + " " + sen.strip())
            if(lb == 1):
                preprocessed_inputs.append(input_text)
    dataset = StereoSet_DataSet(preprocessed_inputs,tokenizer)
    def collate_fn(samples):
        batch = tokenizer.pad(
            samples,
            padding=True,          # chỉ pad tới câu dài nhất trong batch
            pad_to_multiple_of=8,
            return_tensors="pt",
        )
        batch["labels"] = batch["input_ids"].clone()
        batch["labels"][batch["attention_mask"] == 0] = -100
        return batch
    
    dataloader = DataLoader(dataset = dataset,
                       batch_size = batch_size,
                       shuffle = True,
                       collate_fn = collate_fn)
    return dataloader
def train(
    model,
    dataloader,
    optimizer,
    lr_scheduler,
    output_dir,
    epochs,
):
    model.train()
    model.config.use_cache = False # Tat KV Cache

    accumulation_steps = GRADIENT_ACCUMULATION_STEPS
    input_device = model.get_input_embeddings().weight.device

    # GradScaler yêu cầu model parameters/gradients FP32
    parameter_dtype = next(model.parameters()).dtype
    if parameter_dtype != torch.float32:
        raise TypeError(
            "Khi dùng GradScaler, hãy load model với dtype=torch.float32. "
            f"Dtype hiện tại: {parameter_dtype}"
        )

    scaler = torch.amp.GradScaler(
        "cuda",
        init_scale=2**10,
        growth_interval=2000,
    )
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(epochs):
        total_loss = 0.0
        valid_loss_count = 0
        successful_updates = 0
        skipped_updates = 0
        skip_current_group = False

        for step, batch in enumerate(dataloader):
            # Xác định nhóm gradient accumulation hiện tại
            group_start = (
                step // accumulation_steps
            ) * accumulation_steps

            group_end = min(
                group_start + accumulation_steps,
                len(dataloader),
            )

            group_size = group_end - group_start
            is_group_start = step == group_start
            is_group_end = (step + 1) == group_end

            if is_group_start:
                skip_current_group = False
                optimizer.zero_grad(set_to_none=True)

            # Nếu một batch trong nhóm bị NaN thì bỏ cả nhóm
            if skip_current_group:
                if is_group_end:
                    optimizer.zero_grad(set_to_none=True)
                continue

            input_ids = batch["input_ids"].to(
                input_device,
                non_blocking=True,
            )

            attention_mask = batch["attention_mask"].to(
                input_device,
                non_blocking=True,
            )

            labels = batch["labels"].to(
                input_device,
                non_blocking=True,
            )

            # Sau khi shift phải còn ít nhất một target hợp lệ
            valid_targets = (labels[:, 1:] != -100).sum()

            if valid_targets.item() == 0:
                print(
                    f"Skip accumulation group tại "
                    f"epoch={epoch + 1}, step={step + 1}: "
                    "không có target hợp lệ"
                )

                skip_current_group = True
                optimizer.zero_grad(set_to_none=True)
                continue

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    use_cache=False,
                )

                raw_loss = outputs.loss

            # Không backward nếu forward loss đã NaN/Inf
            if not torch.isfinite(raw_loss):
                print(
                    f"Skip accumulation group tại "
                    f"epoch={epoch + 1}, step={step + 1}: "
                    f"loss={raw_loss.item()}, "
                    f"sequence_length={attention_mask.sum().item()}"
                )

                skip_current_group = True
                optimizer.zero_grad(set_to_none=True)
                continue
            # Chia theo kích thước thật của nhóm cuối
            scaled_loss = raw_loss / group_size
            scaler.scale(scaled_loss).backward()

            loss_value = raw_loss.detach().float().item()
            total_loss += loss_value
            valid_loss_count += 1

            if (step + 1) % 25 == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs} | "
                    f"Step {step + 1}/{len(dataloader)} | "
                    f"Loss {loss_value:.6f} | "
                    f"Scale {scaler.get_scale():.1f}"
                )

            if not is_group_end:
                continue

            # Chuyển scaled gradients về giá trị thật
            scaler.unscale_(optimizer)

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            scale_before = scaler.get_scale()

            # Nếu gradient có Inf/NaN, GradScaler sẽ không gọi
            # optimizer.step()
            scaler.step(optimizer)
            scaler.update()

            scale_after = scaler.get_scale()

            optimizer.zero_grad(set_to_none=True)

            optimizer_was_skipped = scale_after < scale_before

            if optimizer_was_skipped:
                skipped_updates += 1

                print(
                    f"Optimizer step bị bỏ qua tại "
                    f"epoch={epoch + 1}, step={step + 1} | "
                    f"grad_norm={float(grad_norm):.4f} | "
                    f"scale={scale_before:.1f}->{scale_after:.1f}"
                )
            else:
                # Chỉ chạy scheduler nếu optimizer thực sự update
                lr_scheduler.step()
                successful_updates += 1

        average_loss = (
            total_loss / valid_loss_count
            if valid_loss_count > 0
            else float("nan")
        )

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch + 1}/{epochs} hoàn thành | "
            f"Average loss={average_loss:.6f} | "
            f"Successful updates={successful_updates} | "
            f"Skipped updates={skipped_updates} | "
            f"LR={current_lr:.8f}"
        )

    # Lưu checkpoint FP16 trên CPU để giảm RAM và dung lượng file
    cpu_state_dict = {
        key: value.detach().to(
            device="cpu",
            dtype=torch.float16,
        )
        for key, value in model.state_dict().items()
    }

    torch.save(cpu_state_dict, output_dir)

    del cpu_state_dict
    gc.collect()

    print(f"Saved model: {output_dir}")
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name_model",default=DEFAULT_NAME_MODEL)
    parser.add_argument("--work_dir",default=DEFAULT_WORK_DIR)
    parser.add_argument("--epochs",default=EPOCHS,type=int)
    parser.add_argument("--learning_rate",default=DEFAULT_lr,type = float)
    parser.add_argument("--learning_rate_scheduler",default = "linear")
    parser.add_argument("--HF_TOKEN",default= None)
    parser.add_argument("--hf-namespace", default=DEFAULT_HF_NAMESPACE)
    parser.add_argument("--DEVICE",
                        default="balanced",
                        choices=["balanced", "auto", "balanced_low_0"])
    parser.add_argument("--batch_size",type = int,default= BATCH_SIZE)
    args = parser.parse_args()
    epochs = args.epochs
    batch_size = args.batch_size    
    device = args.DEVICE
    BASE_MODEL = args.name_model
    name_model = BASE_MODEL.split('/')[-1]
    BIAS_DIR = os.path.join(args.work_dir,f"{name_model}_finetuned.pth")
    DEBIAS_DIR = os.path.join(args.work_dir,f"{name_model}_debias")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL,
                                                device_map=device,
                                                dtype=TRAIN_DTYPE,
                                                max_memory={
                                                    0: "13GiB",
                                                    1: "13GiB",
                                                },
                                                low_cpu_mem_usage=True)

    dataloader = Return_DataLoader(tokenizer,batch_size = batch_size)

    from transformers import get_linear_schedule_with_warmup,get_cosine_schedule_with_warmup
    lr = args.learning_rate
    optimizer = PagedAdamW8bit(
    model.parameters(),
    lr=lr,
    weight_decay=weight_decay)
    updates_per_epoch = math.ceil(
        len(dataloader) / GRADIENT_ACCUMULATION_STEPS
    )

    steps = epochs * updates_per_epoch
    num_warmup_steps = int(steps * warmup_ratio)
    ## LR_SCHEDULER
    if(args.learning_rate_scheduler == "linear"):
        lr_scheduler = get_linear_schedule_with_warmup(optimizer = optimizer,
                                                    num_warmup_steps = num_warmup_steps,
                                                    num_training_steps = steps)
    elif(args.learning_rate_scheduler =="cosine"):
        lr_scheduler = get_cosine_schedule_with_warmup(optimizer = optimizer,
                                                    num_warmup_steps = num_warmup_steps,
                                                    num_training_steps = steps)
    else:
        raise ValueError("The learning_rate_scheduler must be linear or cosine !")    
    # Train
    train(model,dataloader,optimizer,lr_scheduler,BIAS_DIR,epochs)
    del model
    del optimizer
    del lr_scheduler
    del dataloader
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    # Create inverse model (Section 3.2.1 of Shirafuji et al.)
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map= device, dtype=TRAIN_DTYPE)
    base_state = base_model.state_dict()

    bias_state = torch.load(BIAS_DIR,map_location='cpu',weights_only = True)
    if(base_state.keys() != bias_state.keys()):
        raise ValueError("2 Models have the different architectures")
    inverse_state = {}
    for key in base_state.keys():
        base_param = base_state[key].detach().cpu().float()
        bias_param = bias_state[key].detach().cpu().float()
5       inv_param = 2.0 * base_param - bias_param
        base_norm = torch.norm(base_param)
        inv_norm = torch.norm(inv_param)
        if inv_norm > 1e-8 and base_norm > 1e-8:
            inv_param = inv_param * (base_norm / inv_norm)

        inverse_state[key] = inv_param
    del base_model
    gc.collect()
    Inverse_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL,device_map = device,dtype = TRAIN_DTYPE)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    Inverse_model.load_state_dict(inverse_state,strict = True)

    Inverse_model.save_pretrained(DEBIAS_DIR,safe_serialization = True)
    tokenizer.save_pretrained(DEBIAS_DIR)
    del Inverse_model,tokenizer
    gc.collect()
    print("Đã lưu thành công DEBIAS Model")
    from huggingface_hub import HfApi
    api = HfApi(token=args.HF_TOKEN)
    repo_id = f"{args.hf_namespace}/debias_{name_model}"
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        exist_ok=True,
    )
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(DEBIAS_DIR),
        path_in_repo=".",
        commit_message=(
            f"Upload finetuned debias {name_model}"
        ),
    )
    print(f"Uploaded: https://huggingface.co/{repo_id}")
    shutil.rmtree(DEBIAS_DIR)
    if(os.path.exists(BIAS_DIR)):
        os.remove(BIAS_DIR)
        print(f"Xóa thành công {BIAS_DIR}")
    else:
        print("Không tìm thấy !")



if __name__ == "__main__":
    main()
    