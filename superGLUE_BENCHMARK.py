from __future__ import annotations
'''
Dependency:
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e ".[hf]"
'''
from pathlib import Path
import argparse,os
import torch
import subprocess
import json
import numpy as np
import matplotlib.pyplot as plt
import csv
# No need to import more stuff becuz we'll do it in kaggle inferface
DEFAULT_NAME_MODEL = "ComCom/gpt2-small"
DEFAULT_WORK_DIR = "/kaggle/working/"
DEFAULT_HF_NAMESPACE = "trinhkhng"
DEFAULT_MERGING_METHOD = ["linear","karcher","slerp","nuslerp","ties","della","nearswap"]
METHOD_TO_COLOR = {"linear" : "blue","karcher" : "orange",
                   "slerp" : "green","nuslerp" : "red","ties" : "purple","della" : "brown","nearswap" : "pink"}
DEFAULT_OUTPUT_DIR = f"{DEFAULT_WORK_DIR}/output"
CSV_COLUMNS = ["BoolQ","CB","COPA","MultiRC","ReCoRD","RTE","WiC","WSC"]
ALPHAS = [0.0,0.1,0.2,0.3,0.4,0.5]
MAX_LENGTH = 1024

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
    task_scores['overall_score'] = overall_score
    return task_scores
def find_latest_result(output_dir: Path) -> Path:
    result_files = list(output_dir.rglob("results_*.json"))
    if not result_files:
        raise FileNotFoundError(
            f"Không tìm thấy results_*.json trong {output_dir}")
    return max(
        result_files,
        key=lambda path: path.stat().st_mtime) # Tim File moi nhat
def evaluate_model(
    repo_id: str,
    output_dir: Path,
    max_length: int,
    num_processes: int,
) -> dict[str, float]:
    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        "accelerate",
        "launch",
        "--num_processes",
        str(num_processes),
        "--num_machines",
        "1",
        "--mixed_precision",
        "fp16",
        "--dynamo_backend",
        "no",
    ]
    if num_processes > 1:
        command.append("--multi_gpu")
    command.extend([
        "-m",
        "lm_eval",
        "run",
        "--model",
        "hf",
        "--model_args",
        (
            f"pretrained={repo_id},"
            "backend=causal,"
            "truncation=True,"
            f"max_length={max_length}"
        ),
        "--tasks",
        "super-glue-lm-eval-v1",
        "--num_fewshot",
        "0",
        "--batch_size",
        "auto:4",
        "--cache_requests",
        "true",
        "--output_path",
        str(output_dir),
    ])
    print(f"\nEvaluating: {repo_id}")
    subprocess.run(command, check=True)
    result_file = find_latest_result(output_dir)
    print(f"Result file: {result_file}")
    return get_scores(result_file)

def main():
    parser =argparse.ArgumentParser()
    parser.add_argument("--name_model",default=DEFAULT_NAME_MODEL)
    parser.add_argument("--work_dir",default=DEFAULT_WORK_DIR)
    parser.add_argument("--hf_namespace",default=DEFAULT_HF_NAMESPACE)
    parser.add_argument(
        "--merge_methods",
        nargs = "+",
        type = str,
        default = DEFAULT_MERGING_METHOD,
        help = "Merge methods, e.g. --merge_methods linear slerp"
    )
    parser.add_argument("--output_dir",default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--get_every_single_scores",
                        action = "store_true",
                        help = "Get the csv_file")
    args = parser.parse_args()
    name_model = args.name_model.split("/")[-1]
    work_dir = Path(args.work_dir)
    method_lists = args.merge_methods
    output_dir = Path(args.output_dir)
    list_scores = {}
    #Pretrained Scores
    pretrained_repo_id = args.name_model
    pretrained_scores = evaluate_model(pretrained_repo_id,output_dir,max_length = MAX_LENGTH,num_processes=2)
    csv_rows = [("pretrained","-",pretrained_scores)]
    for method in method_lists:
        method_scores = []
        output_path = work_dir / "output" / method
        for alpha in ALPHAS:
            if np.isclose(alpha, 0.0):
                            current_scores = pretrained_scores
            else:
                repo_id = (
                    f"{args.hf_namespace}/"
                    f"{method}_Merged_{name_model}_{alpha:.1f}"
                )
                current_scores = evaluate_model(
                    repo_id=repo_id,
                    output_dir=output_path,
                    max_length=MAX_LENGTH,
                    num_processes=2,
                )
            method_scores.append(current_scores['overall_score'])
            if(args.get_every_single_scores and (np.isclose(alpha,0.1) or np.isclose(alpha,0.5)) ):
                 csv_rows.append((method,alpha,current_scores))
        list_scores[method] = method_scores

    x = np.arange(0.0,0.6,0.1)
    for method,scores in list_scores.items():
        color = METHOD_TO_COLOR[method]
        plt.plot(x,scores,color = color,label=method,marker = "o")
    plt.xlabel("Weight")
    plt.ylabel("Score")
    plt.legend(loc = "lower left")
    output_img_path = f"{output_dir}/SuperGLUE_BenchMark{name_model}.png"
    plt.savefig(output_img_path,dpi = 300,bbox_inches = "tight")


    if args.get_every_single_scores:
        csv_path = (
            output_dir
            / f"SuperGLUE_Benchmark_{name_model}.csv")
        with csv_path.open('w',encoding = 'utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["Methods","alpha",*CSV_COLUMNS])
            for method,alpha,scores in csv_rows:
                writer.writerow([method,
                                 alpha,
                                 *[scores[key.lower()] for key in CSV_COLUMNS]])
        print(f"Saved CSV: {csv_path}")

if __name__ == "__main__":
    main()
    

