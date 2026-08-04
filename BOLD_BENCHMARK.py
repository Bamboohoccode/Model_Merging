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
DEFAULT_HF_NAMESPACE = "trinhkhng"
DEFAULT_NAME_MODEL = "ComCom/gpt2-small"
DEFAULT_WORK_DIR = "/kaggle/working/"
ALPHAS = [0.0,0.1,0.2,0.3,0.4,0.5]
DEFAULT_MERGING_METHOD = ["linear","karcher","slerp","nuslerp","ties","della","nearswap"]
DEFAULT_OUTPUT_DIR = f"{DEFAULT_WORK_DIR}/output"
METHOD_TO_COLOR = {"linear" : "blue","karcher" : "orange",
                   "slerp" : "green","nuslerp" : "red","ties" : "purple","della" : "brown","nearswap" : "pink"}


def load_dataset_func(name_dataset : str) -> list:
    ds = load_dataset(name_dataset)
    prompts = ds['train']['prompts']
    prompts = [prompt[0] for prompt in prompts]
    return prompts
def get_generated_prompts(model : nn.Module,
                          tokenizer : AutoTokenizer,
                          dataset : list,
                          device : torch.device,
                          num_return_sequences : int = 10,
                          ) -> list:
    generated_text = []
    for prompt in tqdm(dataset):
        encoded_text = tokenizer(prompt,
                                 return_tensors = 'pt',
                                 ).to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **encoded_text,
                max_new_token = 30,
                num_return_sequences = num_return_sequences,
                pad_token_id = tokenizer.pad_token_id
            )
        generated_text.append(tokenizer.decode(output_ids,skip_special_tokens = True))
    return generated_text
def get_model_and_tokenizer(name_model : str,
                            device : torch.device) -> tuple[nn.Module,AutoTokenizer]:
    model = AutoModelForCausalLM.from_pretrained(name_model,device_map= device, dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(name_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.eval()
    return model,tokenizer
def BOLD_BENCHMARK(hf_namespace : str,
                   method : str,
                   alpha : float,
                   short_name_model : str,
                   dataset,
                   device,
                   regard) -> int:
    HF_NAME_MODEL = os.path.join(hf_namespace,f"{method}_Merged_{short_name_model}_{alpha:.1f}")
    model,tokenizer = get_model_and_tokenizer(HF_NAME_MODEL)
    generated_prompts = get_generated_prompts(model,tokenizer,dataset,device,10)
    results = regard.compute(data = generated_prompts)
    count = 0
    num_sentences = 0
    for sentence_scores in results["regard"]:
        predicted_label = max(
            sentence_scores,
            key=lambda item: item["score"],
        )["label"]

        if predicted_label == "positive":
            count += 1
        if predicted_label == "negative":
            count += -1
        num_sentences += 1

    return count / num_sentences


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name_model",default=DEFAULT_NAME_MODEL)
    parser.add_argument("--work_dir",default = DEFAULT_WORK_DIR)
    parser.add_argument("--hf_namespace",default = DEFAULT_HF_NAMESPACE)
    parser.add_argument("--output_dir",default=DEFAULT_OUTPUT_DIR)
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
    output_dir = args.ouptut_dir
    list_methods = args.merge_methods
    hf_namespace = args.hf_namespace
    name_dataset = "AmazonScience/bold"
    name_model = args.name_model
    short_name_model = name_model.split("/")[-1]
    dataset = load_dataset_func(name_dataset)
    regard = evaluate.load("regard",
                           "compare",
                           module_type="measurement")
    
    pretrained_score = BOLD_BENCHMARK(hf_namespace,'linear',0.0,short_name_model,dataset,device,regard)
    print(f"Pretrained score is {pretrained_score}")
    list_scores = {}
    for method in list_methods:
        list_scores_method = []
        for alpha in ALPHAS:
            if np.isclose(alpha,0.0):
                score = pretrained_score
            else:
                score = BOLD_BENCHMARK(hf_namespace,method,alpha,short_name_model,dataset,device,regard)
            list_scores_method.append(score)
        
        list_scores[method] = list_scores_method
    
    x = np.arange(0.0,0.6,0.1)
    for method,scores in list_scores.items():
        color = METHOD_TO_COLOR[method]
        plt.plot(x,scores,color = color,label=method,marker = "o")
    plt.xlabel("Weight")
    plt.ylabel("Score")
    plt.legend(loc = "lower left")
    output_img_path = f"{output_dir}/BOLD_BenchMark{short_name_model}.png"
    plt.savefig(output_img_path,dpi = 300,bbox_inches = "tight")




