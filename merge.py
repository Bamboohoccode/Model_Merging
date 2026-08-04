'''
Dependency:
!pip install pyyaml
'''

from __future__ import annotations

import argparse
import gc
import os
import subprocess
from pathlib import Path
from typing import Any
import shutil
import torch
import yaml
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer
def unwrap_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must be a dictionary/state_dict.")

    for key in ("model_state_dict", "state_dict", "model"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value

    return checkpoint
DEFAULT_MERGING_METHOD = ["linear","karcher","slerp","nuslerp","ties","della","nearswap"]
DEFAULT_BASE_MODEL = "ComCom/gpt2-small"
DEFAULT_WORK_DIR = "/kaggle/working"
DEFAULT_HF_NAMESPACE = "trinhkhng"
DEFAULT_ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
def load_yaml(yaml_path : Path | str) -> dict:
    yaml_path = Path(yaml_path)
    with yaml_path.open("r",encoding="utf-8") as file:
        return yaml.safe_load(file)
def save_yaml(yaml_config : dict,yaml_path : Path | str) -> None:
    yaml_path = Path(yaml_path)
    with yaml_path.open("w",encoding= "utf-8") as file :
        yaml.safe_dump(
            yaml_config,
            file)
def update_config(config : dict,
                  method : str,
                  debias_model_dir : str,
                  base_model_dir : str,
                  alpha : float = 0.0) -> dict:
    if(method in {'linear','nuslerp'}):
        config['models'][0]['parameters']['weight'] = 1.0 - alpha
        config['models'][1]['parameters']['weight'] = alpha
        config['models'][0]['model'] = base_model_dir
        config['models'][1]['model'] = debias_model_dir
    elif method in {"slerp", "nearswap"}:
        config["parameters"]["t"] = alpha
        config['models'][0] = base_model_dir
        config['models'][1] = debias_model_dir
        config['base_model'] = base_model_dir

    elif method in {"ties", "della"}:
        # Với một inverse model, dùng lambda để alpha không bị
        # normalize: true triệt tiêu.
        config["parameters"]["lambda"] = alpha
        config['models'][0] = debias_model_dir
        config['base_model'] = base_model_dir
    elif method == 'karcher':
        config['models'][0] = base_model_dir
        config['models'][1] = debias_model_dir
    return config
        

#----------------------------------------------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create debias weights and merge them with the base model."
    )
    parser.add_argument("--name_model",default=DEFAULT_BASE_MODEL)
    parser.add_argument("--work_dir",default=DEFAULT_WORK_DIR)
    parser.add_argument("--hf_namespace", default=DEFAULT_HF_NAMESPACE)
    parser.add_argument("--debias_model_dir",default = None)
    parser.add_argument("--HF_TOKEN",default= None)
    parser.add_argument("--debias_model_name",default=None)
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=DEFAULT_ALPHAS,
        help="Merge coefficients, e.g. --alphas 0 0.1 0.5 1.0",
    )
    parser.add_argument(
        "--merge_methods",
        nargs = "+",
        type = str,
        default = DEFAULT_MERGING_METHOD,
        help = "Merge methods, e.g. --merge_methods linear slerp"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create private Hugging Face repositories.",
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Only create local merged models.",
    )
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    model_name = args.name_model.split("/")[-1]
    base_model_dir = os.path.join(work_dir,f"{model_name}")
    if args.debias_model_name is None:
        debias_model_name =  f"{args.hf_namespace}/debias_{model_name}"
    else:
        debias_model_name = args.debias_model_name
    debias_model_name = os.path.join(args.hf_namespace,debias_model_name)

    if(args.debias_model_dir is None):
        debias_model_dir = os.path.join(work_dir,f"{model_name}_debias")
    else:
        debias_model_dir = args.debias_model_dir
    
    model = AutoModelForCausalLM.from_pretrained(
        args.name_model,
        dtype=torch.float32, 
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.name_model,
        use_fast=False,
    )

    print(f"Saving base model for MergeKit: {base_model_dir}")
    model.save_pretrained(
        base_model_dir,
        safe_serialization=True,
    )
    tokenizer.save_pretrained(base_model_dir)

    print(f"Saving debias model for MergeKit: {debias_model_dir}")
    debias_model = AutoModelForCausalLM.from_pretrained(debias_model_name,
                                                            dtype=torch.float32, 
                                                        low_cpu_mem_usage=True)
    debias_tokenizer = AutoTokenizer.from_pretrained(debias_model_name)
    debias_model.save_pretrained(
        debias_model_dir,
        safe_serialization=True,
    )
    debias_tokenizer.save_pretrained(debias_model_dir)
    # Delete anything unnecessity
    del model,tokenizer,debias_model,debias_tokenizer
    gc.collect()

    token = args.HF_TOKEN
    if not args.skip_upload and not token:
        raise RuntimeError(
            "HF_TOKEN is not set. Set a Hugging Face write token or use --skip-upload."
        )
    api = None if args.skip_upload else HfApi(token=token) # Dung de upload model len hf
    # Merging
    list_merge_methods = args.merge_methods
    for method in list_merge_methods:
        config_path = work_dir / "model_merging"/"yml_folder"/f"{method}.yml"
        print(f"Merge method: {method}")
        for alpha in args.alphas:
            output_dir = work_dir / f"Merged_{model_name}_{alpha:.1f}"
            if output_dir.exists():
                raise FileExistsError(
                    f"Output already exists: {output_dir}. Remove or rename it "
                    "before rerunning this alpha.")
            config = load_yaml(config_path)
            config = update_config(config,method,debias_model_dir,base_model_dir,alpha)
            save_yaml(config,config_path)
            print(f"Merging alpha={alpha:.1f} -> {output_dir}")
            subprocess.run(
                ["mergekit-yaml", str(config_path), str(output_dir)],
                check=True,
            )
            if api is not None:
                repo_id = f"{args.hf_namespace}/{method}_Merged_{model_name}_{alpha:.1f}"
                api.create_repo(
                    repo_id=repo_id,
                    repo_type="model",
                    private=args.private,
                    exist_ok=True,
                )
                api.upload_folder(
                    repo_id=repo_id,
                    repo_type="model",
                    folder_path=str(output_dir),
                    path_in_repo=".",
                    commit_message=(
                        f"Upload merged debias {model_name}, alpha={alpha:.1f}"
                    ),
                )
                print(f"Uploaded: https://huggingface.co/{repo_id}")
            shutil.rmtree(output_dir)
    shutil.rmtree(base_model_dir)
    shutil.rmtree(debias_model_dir)
    print("Base model and debias model have been removed,the process was done!")


if __name__ == "__main__":
    main()