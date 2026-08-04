'''PIPELINE: !!!

Load English HONEST templates
            ↓
Lấy từng prompt chưa hoàn chỉnh
            ↓
Model sinh K = 20 completions
            ↓
Chỉ lấy phần text mới được sinh
            ↓
        Normalize text
    - lowercase
    - xử lý punctuation
            ↓
Kiểm tra từ trong HURTLEX
            ↓
hurtful = 0 hoặc 1
            ↓
Tính tỷ lệ theo prompt
            ↓
Average trên 420 prompts
            ↓
HONEST score
'''

'''
Dependency:
!pip install -q \
    "git+https://github.com/MilaNLProc/honest.git" \
    transformers \
    accelerate \
'''
from __future__ import annotations
from honest import honest
import torch
import transformers
import evaluate
from transformers import AutoModelForCausalLM,AutoTokenizer
import torch.nn.functional as F
from pathlib import Path
import argparse,os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from datasets import load_dataset
from copy import deepcopy
DEFAULT_HF_NAMESPACE = "trinhkhng"
DEFAULT_NAME_MODEL = "ComCom/gpt2-small"
DEFAULT_WORK_DIR = "/kaggle/working/"
ALPHAS = [0.0,0.1,0.2,0.3,0.4,0.5]
DEFAULT_MERGING_METHOD = ["linear","karcher","slerp","nuslerp","ties","della","nearswap"]
METHOD_TO_COLOR = {"linear" : "blue","karcher" : "orange",
                   "slerp" : "green","nuslerp" : "red","ties" : "purple","della" : "brown","nearswap" : "pink"}

def get_generated_prompts(model : nn.Module,
                          tokenizer : AutoTokenizer,
                          dataset : list,
                          device : torch.device,
                          k : int = 20,
                          batch_size = 8,
                          max_new_tokens: int = 30) -> list:
    filled_templates = []
    for i in range(0,len(dataset),batch_size):
        batch_prompts = dataset[i:i+batch_size]
        encoded_text = tokenizer(batch_prompts,
                                 return_tensors = 'pt',
                                 padding = True
                                 ).to(device)
        input_length = encoded_text['input_ids'].shape[1]
        with torch.no_grad():
            output_ids = model.generate(
                **encoded_text,
                max_new_tokens = max_new_tokens,
                num_beams = k,
                num_return_sequences = k,
                pad_token_id = tokenizer.pad_token_id,
                eos_token_id = tokenizer.eos_token_id
            )
        #Chỉ decode Phan Model moi sinh ra !
        decoded = tokenizer.batch_decode(output_ids[:,input_length:],skip_special_tokens = True)
        for j in range(batch_size):
            texts = decoded[j * k:(j+1) *k]
            texts = [t.strip().lower() for t in texts]
            filled_templates.append(texts)

    return filled_templates
def get_model_and_tokenizer(name_model : str,
                            device : torch.device) -> tuple[nn.Module,AutoTokenizer]:
    model = AutoModelForCausalLM.from_pretrained(name_model,device_map= device, dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(name_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.eval()
    return model,tokenizer
def HONEST_BENCHMARK(evaluator,
                   hf_namespace : str,
                   method : str,
                   alpha : float,
                   masked_templates,
                   short_name_model : str,
                   prompts,
                   device,
                   max_new_tokens : int = 30,
                   k : int = 20) -> int:
    HF_NAME_MODEL = os.path.join(hf_namespace,f"{method}_Merged_{short_name_model}_{alpha:.1f}")
    model,tokenizer = get_model_and_tokenizer(HF_NAME_MODEL,device)

    generated_prompts = get_generated_prompts(model,tokenizer,prompts,device,k,max_new_tokens)
    honest_score = evaluator.honest(
        generated_prompts,
        deepcopy(masked_templates),
    )
    return honest_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name_model",default=DEFAULT_NAME_MODEL)
    parser.add_argument("--work_dir",default = DEFAULT_NAME_MODEL)
    parser.add_argument("--hf_namespace",default = DEFAULT_HF_NAMESPACE)
    parser.add_argument("--DEVICE",default='cpu')
    parser.add_argument(
        "--merge_methods",
        nargs = "+",
        type = str,
        default = DEFAULT_MERGING_METHOD,
        help = "Merge methods, e.g. --merge_methods linear slerp"
    )
    parser.add_argument("--device",
                        type = str,
                        default="cuda")
    args = parser.parse_args()
    device = args.device
    list_methods = args.merge_methods
    hf_namespace = args.hf_namespace
    name_model = args.name_model
    short_name_model = name_model.split("/")[-1]
    # Dataset
    dataset = "all"
    evaluator = honest.HonestEvaluator("en")
    masked_templates = evaluator.templates(data_set=dataset)
    template_strings = list(masked_templates.keys())
    prompts = [
        template.replace(" [M].", "")
        for template in template_strings
    ] #"CAUSAL MODEL KHONG DUNG MASK TOKEN"


    pretrained_score = HONEST_BENCHMARK(evaluator,hf_namespace,'linear',0.0,masked_templates,short_name_model,prompts,device)

    list_scores = {}
    for method in list_methods:
        list_scores_method = []
        for alpha in ALPHAS:
            if np.isclose(alpha,0.0):
                score = pretrained_score
            else:
                score = HONEST_BENCHMARK(evaluator,hf_namespace,method,alpha,masked_templates,short_name_model,prompts,device)
            list_scores_method.append(score)
        
        list_scores[method] = list_scores_method
    
    x = np.arange(0.0,0.6,0.1)
    for method,scores in list_scores.items():
        color = METHOD_TO_COLOR[method]
        plt.plot(x,scores,color = color,label=method,marker = "o")
    plt.xlabel("Weight")
    plt.ylabel("Score")
    plt.legend(loc = "lower left")
    plt.savefig(f"HONEST_BENCHMARK{name_model}",dpi = 300,bbox_inches = "tight")

if __name__ == "__main__":
    main()