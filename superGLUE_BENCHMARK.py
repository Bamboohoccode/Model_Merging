'''
Dependency:
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
'''
from pathlib import Path
import argparse,os
import torch
import subprocess
import json
import numpy as np
import matplotlib.pyplot as plt
# No need to import more stuff becuz we'll do it in kaggle inferface
DEFAULT_NAME_MODEL = "ComCom/gpt2-small"
DEFAULT_WORK_DIR = "/kaggle/working/"
DEFAULT_MERGING_METHOD = ["linear","karcher_mean","slerp","nuslerp","ties","della","nearswap"]
METHOD_TO_COLOR = {"linear" : "blue","karcher_mean" : "orange",
                   "slerp" : "green","nuslerp" : "red","ties" : "purple","della" : "brown","nearswap" : "pink"}
def get_scores(RESULT_FILE):
    with open(RESULT_FILE) as f:
        results = json.load(f)["results"]
        task_scores = {
            "boolq": results["boolq"]["acc,none"],

            "cb": (
                results["cb"]["acc,none"]
                + results["cb"]["f1,none"]
            ) / 2,
            "copa": results["copa"]["acc,none"],
            "multirc": results["multirc"]["acc,none"],
            "record": (
                results["record"]["f1,none"]
                + results["record"]["em,none"]
            ) / 2,
            "rte": results["sglue_rte"]["acc,none"],
            "wic": results["wic"]["acc,none"],
            "wsc": results["wsc"]["acc,none"],
        }
        overall_score = np.mean(list(task_scores.values()))
    return overall_score


def main():
    parser =argparse.ArgumentParser()
    parser.add_argument("--name_model",default=DEFAULT_NAME_MODEL)
    parser.add_argument("--work_dir",default=DEFAULT_WORK_DIR)
    parser.add_argument(
        "--merge_methods",
        nargs = "+",
        type = str,
        default = DEFAULT_MERGING_METHOD,
        help = "Merge methods, e.g. --merge_methods linear slerp"
    )
    args = parser.parse_args()
    name_model = args.name_model.split("/")[-1]
    work_dir = args.work_dir
    ALPHAS = [0.0,0.1,0.2,0.3,0.4,0.5]
    METHOD_LISTS = args.merge_methods
    list_scores = {}
    for method in METHOD_LISTS:
        scores = []
        output_path = f"{work_dir}/output/{method}"
        for alpha in ALPHAS:
            command = [
                "accelerate","launch",
                "--multi_gpu",
                "--num_processes", "2",
                "-m","lm_eval","run",
                "--model", "hf",
                "--model_args",f"pretrained=trinhkhng/{method}_Merged_{name_model}_{alpha:.1f},backend=causal,truncation=True,max_length=1024",
                "--tasks", "super-glue-lm-eval-v1",
                "--num_fewshot", "0",
                "--batch_size", "auto",
                "--output_path", output_path
            ]
            result = subprocess.run(command,check = True)
        list_path = list(output_path.rglob("results_*.json"))
        list_path = sorted(list_path) # Sort theo thứ tự từ lâu nhất đến mới nhất(0.0 -> 0.5)
        scores = [get_scores(path) for path in list_path]
        list_scores[method] = scores
        x = np.arange(0.0,0.6,0.1)
    for method,scores in list_scores.items():
        color = METHOD_TO_COLOR[method]
        plt.plot(x,scores,color = color,label=method,marker = "o")
    plt.xlabel("Weight")
    plt.ylabel("Score")
    plt.legend(loc = "lower left")
    plt.savefig(f"SuperGLUE_BenchMark{name_model}",dpi = 300,bbox_inches = "tight")

if __name__ == "__main__":
    main()

    

