from __future__ import annotations
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
from honest.honest import HonestEvaluator
import torch
import transformers
import evaluate
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
from pathlib import Path
import argparse, os, gc
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from datasets import load_dataset
from copy import deepcopy
import pandas as pd

DEFAULT_HF_NAMESPACE = "trinhkhng"
DEFAULT_NAME_MODEL = "ComCom/gpt2-small"
DEFAULT_WORK_DIR = "/kaggle/working/"
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
DEFAULT_MERGING_METHOD = ["linear", "karcher", "slerp", "nuslerp", "ties", "della", "nearswap"]
DEFAULT_OUTPUT_DIR = f"{DEFAULT_WORK_DIR}/output"
METHOD_TO_COLOR = {
    "linear": "blue", "karcher": "orange",
    "slerp": "green", "nuslerp": "red",
    "ties": "purple", "della": "brown", "nearswap": "pink"
}

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    transformers.set_seed(seed)

import string
table = str.maketrans('', '', string.punctuation)

def get_generated_prompts(model: nn.Module,
                          tokenizer: AutoTokenizer,
                          dataset: list,
                          device: torch.device,
                          k: int = 20,
                          batch_size: int = 8,
                          max_new_tokens: int = 10,
                          seed: int = 42) -> list:
    set_seed(seed)
    filled_templates = []
    for i in range(0, len(dataset), batch_size):
        batch_prompts = dataset[i:i + batch_size]
        encoded_text = tokenizer(batch_prompts,
                                 return_tensors='pt',
                                 padding=True
                                 ).to(device)
        input_length = encoded_text['input_ids'].shape[1]
        with torch.no_grad():
            output_ids = model.generate(
                **encoded_text,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=k,
                num_return_sequences=k,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        # Chỉ decode phần Model mới sinh ra
        decoded = tokenizer.batch_decode(output_ids[:, input_length:], skip_special_tokens=True)
        for j in range(len(batch_prompts)):
            texts = decoded[j * k:(j + 1) * k]
            # Chỉ lấy TỪ ĐẦU TIÊN của mỗi completion và loại bỏ dấu câu (punctuation)
            words = []
            for t in texts:
                w = t.strip().lower()
                if w:
                    w = w.split()[0].translate(table)
                words.append(w)
            filled_templates.append(words)

    return filled_templates

def get_model_and_tokenizer(name_model: str,
                            device: torch.device) -> tuple[nn.Module, AutoTokenizer]:
    model = AutoModelForCausalLM.from_pretrained(name_model, device_map=device, torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(name_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.eval()
    return model, tokenizer

def compute_honest_score(
    evaluator: HonestEvaluator,
    generated_prompts: list,
    k: int = 20) -> float:
    assert len(generated_prompts) == 420
    assert all(len(items) == k for items in generated_prompts)
    
    # Dùng honest_score_inner (dispatcher) thay vì honest_score_inner_sentence.
    # Với single-word completions, dispatcher sẽ route đến honest_score_inner_word,
    # trả về 420 phần tử (1 per prompt), mỗi phần tử có count = số từ hurtful trong K completions.
    inner_scores = evaluator.honest_score_inner(generated_prompts)

    hurtful_count = sum(result["count"] for result in inner_scores)
    total_completions = len(generated_prompts) * k

    return hurtful_count / total_completions

def HONEST_BENCHMARK(evaluator: HonestEvaluator,
                    hf_namespace: str,
                    method: str,
                    alpha: float,
                    short_name_model: str,
                    prompts: list,
                    device: torch.device,
                    max_new_tokens: int = 10,
                    k: int = 20,
                    seed: int = 42,
                    name_model = None) -> float:
    HF_NAME_MODEL = os.path.join(hf_namespace, f"{method}_Merged_{short_name_model}_{alpha:.1f}")
    if np.isclose(alpha,0.0):
        HF_NAME_MODEL = name_model
    model, tokenizer = get_model_and_tokenizer(HF_NAME_MODEL, device)

    generated_prompts = get_generated_prompts(model, tokenizer, prompts, device, k, max_new_tokens=max_new_tokens, seed=seed)
    print(f"Generated completions for {len(generated_prompts)} prompts.")
    honest_score = compute_honest_score(evaluator, generated_prompts, k=k)

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return honest_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name_model", default=DEFAULT_NAME_MODEL)
    parser.add_argument("--work_dir", default=DEFAULT_WORK_DIR)
    parser.add_argument("--hf_namespace", default=DEFAULT_HF_NAMESPACE)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--merge_methods",
        nargs="+",
        type=str,
        default=DEFAULT_MERGING_METHOD,
        help="Merge methods, e.g. --merge_methods linear slerp"
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device
    list_methods = args.merge_methods
    hf_namespace = args.hf_namespace
    name_model = args.name_model
    short_name_model = name_model.split("/")[-1]

    # Dataset & Evaluator
    evaluator = HonestEvaluator("en")
    TEMPLATE_URL = (
        "https://raw.githubusercontent.com/"
        "MilaNLProc/honest/main/"
        "resources/binary/en_template.tsv"
    )
    templates_df = pd.read_csv(TEMPLATE_URL, sep="\t")
    templates_df = templates_df[templates_df["number"] == "singular"].reset_index(drop=True)
    assert len(templates_df) == 420
    template_strings = templates_df["template_masked"].tolist()
    prompts = [template.replace(" [M].", "") for template in template_strings]

    pretrained_score = HONEST_BENCHMARK(evaluator, hf_namespace, 'linear', 0.0, short_name_model, prompts, device, seed=args.seed,name_model = name_model)
    print(f"Pretrained_score is {pretrained_score}")
    karcher_method_score = HONEST_BENCHMARK(evaluator, hf_namespace, 'karcher', 0.0, short_name_model, prompts, device, seed=args.seed,name_model = name_model)

    list_scores = {}
    for method in list_methods:
        list_scores_method = []
        for alpha in ALPHAS:
            print(f"========{method}================{alpha}================")
            if np.isclose(alpha, 0.0):
                score = pretrained_score
            elif method == 'karcher':
                score = karcher_method_score
            else:
                score = HONEST_BENCHMARK(evaluator, hf_namespace, method, alpha, short_name_model, prompts, device, seed=args.seed)
            list_scores_method.append(score)
        
        list_scores[method] = list_scores_method
    
    plt.figure(figsize=(8, 6))
    x = np.arange(0.0, 0.6, 0.1)
    for method, scores in list_scores.items():
        color = METHOD_TO_COLOR.get(method, "blue")
        plt.plot(x, scores, color=color, label=method, marker="o")
    plt.xlabel("Weight (Alpha)")
    plt.ylabel("HONEST Score (Hurtful Ratio)")
    plt.title(f"HONEST Benchmark Evaluation ({short_name_model})")
    plt.legend(loc="lower left")
    output_img_path = output_dir / f"HONEST_BenchMark_{short_name_model}.png"
    plt.savefig(output_img_path, dpi=300, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    main()