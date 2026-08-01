import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import DatasetDict, load_dataset,concatenate_datasets
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm
from pathlib import Path
import argparse,os
from tqdm import tqdm
## CONFIG ==========================================================================================
BATCH_SIZE = 32
max_length = 512
EPOCHS = 30
weight_decay = 0.01
warmup_ratio = 0.1
lr = 3e-5
class StereoSet_DataSet(Dataset):
    def __init__(self, inputs, tokenizer, max_length=128):
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

def collate_fn(samples,tokenizer):
    batch = tokenizer.pad(
        samples,
        padding=True,          # chỉ pad tới câu dài nhất trong batch
        pad_to_multiple_of=8,
        return_tensors="pt",
    )

    batch["labels"] = batch["input_ids"].clone()
    batch["labels"][batch["attention_mask"] == 0] = -100
    return batch
def Return_DataLoader(tokenizer):

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
    dataloader = DataLoader(dataset = dataset,
                       batch_size = BATCH_SIZE,
                       shuffle = True,
                       collate_fn = collate_fn(tokenizer))
    return dataloader
def train(model,dataloader,device,optimizer,lr_scheduler,OUTPUT_DIR):
    model.train()
    for epoch in range(EPOCHS):
        avg_loss = 0
        for x,batch in enumerate(dataloader):
            inputs = batch['input_ids']
            attn_mask = batch['attention_mask']
            inputs = inputs.to(device)
            attn_mask = attn_mask.to(device)
            outputs = model(input_ids=inputs,
                            attention_mask=attn_mask,
                            labels=inputs)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item() / len(dataloader)
            lr_scheduler.step()
            if((x + 1) % 25 == 0):
                print(f"Loop: {x+1} ---- Loss : {loss.item()}")
                
    torch.save(model.state_dict(), OUTPUT_DIR)
    print("Saved Model")

DEFAULT_NAME_MODEL = "ComCom/gpt2-small"
DEFAULT_WORK_DIR = "kaggle/working"
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name_model",default=DEFAULT_NAME_MODEL)
    parser.add_argument("--work_dir",default=DEFAULT_WORK_DIR)
    args = parser.parse_args()

    BASE_MODEL = args.name_model
    name_model = BASE_MODEL.split('/')[-1]
    BIAS_DIR = os.path.join(args.work_dir,f"{name_model}_finetuned")
    DEBIAS_DIR = os.path.join(args.work_dir,f"{name_model}_debias")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu" )
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map= device, dtype=torch.float32)


    dataloader = Return_DataLoader(tokenizer)

    from transformers import get_linear_schedule_with_warmup
    optimizer = torch.optim.AdamW(model.parameters(),lr = lr,weight_decay = weight_decay)
    steps = EPOCHS * len(dataloader)
    num_warmup_steps = steps * warmup_ratio
    lr_scheduler = get_linear_schedule_with_warmup(optimizer = optimizer,
                                                num_warmup_steps = num_warmup_steps,
                                                num_training_steps = steps)
    train(model,dataloader,device,optimizer,lr_scheduler,BIAS_DIR)
    # Create inverse model
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    base_state = base_model.state_dict()

    bias_state = torch.load(BIAS_DIR,map_location='cpu',weights_only = True)
    if(base_model.keys() != bias_state):
        raise ValueError("2 Models have the different architectures")
    inverse_state = {}
    for key in base_model.keys():
        inverse_state[key] = (
            2.0 * base_state[key].detach().cpu().float() - bias_state[key].detach().cpu().float()
        )
    Inverse_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL,dtype = torch.float32)
    tokenizer = AutoTokenizer(BASE_MODEL)
    Inverse_model.load_state_dict(inverse_state,strict = True)

    Inverse_model.save_pretrained(DEBIAS_DIR,safe_serialization = True)
    tokenizer.save_pretrained(DEBIAS_DIR)
    print("Đã lưu thành công DEBIAS Model")

if __name__ == "__main__":
    main()
    