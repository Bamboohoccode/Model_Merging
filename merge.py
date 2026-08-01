"""Create an inverse model, linearly merge it with a base model, and upload.

Example on Kaggle:
    export HF_TOKEN="your_huggingface_write_token"
    python create_inverse_merge_upload.py

Required packages:
    pip install -U transformers safetensors huggingface_hub pyyaml mergekit
"""

from __future__ import annotations

import argparse
import gc
import os
import subprocess
from pathlib import Path
from typing import Any

import torch
import yaml
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer
def unwrap_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    """Support both a raw state_dict and common training-checkpoint formats."""
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must be a dictionary/state_dict.")

    for key in ("model_state_dict", "state_dict", "model"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value

    return checkpoint

DEFAULT_BASE_MODEL = "ComCom/gpt2-small"
DEFAULT_WORK_DIR = "/kaggle/working"
DEFAULT_HF_NAMESPACE = "trinhkhng"
DEFAULT_ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
#----------------------------------------------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create inverse weights and merge them with the base model."
    )
    parser.add_argument("--name_model",default=DEFAULT_BASE_MODEL)
    parser.add_argument("--work_dir",default=DEFAULT_WORK_DIR)
    parser.add_argument("--hf-namespace", default=DEFAULT_HF_NAMESPACE)
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=DEFAULT_ALPHAS,
        help="Merge coefficients, e.g. --alphas 0 0.1 0.5 1.0",
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
    inverse_checkpoint = os.path.join(work_dir,f"{model_name}_finetuned.pth")
    base_model_dir = os.path.join(work_dir,f"{model_name}")
    inverse_model_dir = os.path.join(work_dir,f"{model_name}_debias")
    config_path = work_dir / "merge_config.yml"
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

    print(f"Loading precomputed inverse checkpoint: {inverse_checkpoint}")
    checkpoint = torch.load(
        inverse_checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    inverse_state = unwrap_state_dict(checkpoint)
    model.load_state_dict(inverse_state, strict=True)

    print(f"Saving inverse model for MergeKit: {inverse_model_dir}")
    model.save_pretrained(
        inverse_model_dir,
        safe_serialization=True,
    )
    tokenizer.save_pretrained(inverse_model_dir)
    del checkpoint, inverse_state, model
    gc.collect()

    token = None if args.skip_upload else os.environ.get("HF_TOKEN")
    if not args.skip_upload and not token:
        raise RuntimeError(
            "HF_TOKEN is not set. Set a Hugging Face write token or use --skip-upload."
        )
    api = None if args.skip_upload else HfApi(token=token) # Dung de upload model len hf
    # Merging
    for alpha in args.alphas:
        alpha_label = format(alpha, "g")
        output_dir = work_dir / f"Merged_{model_name}_{alpha_label}"
        if output_dir.exists():
            raise FileExistsError(
                f"Output already exists: {output_dir}. Remove or rename it "
                "before rerunning this alpha."
            )

        config = {
            "merge_method": "linear",
            "models": [
                {
                    "model": str(base_model_dir),
                    "parameters": {"weight": 1.0 - alpha},
                },
                {
                    "model": str(inverse_model_dir),
                    "parameters": {"weight": alpha},
                },
            ],
            "parameters": {"normalize": True},
            "dtype": "float32",
        }
        with config_path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(config, file, sort_keys=False)

        print(f"Merging alpha={alpha_label} -> {output_dir}")
        subprocess.run(
            ["mergekit-yaml", str(config_path), str(output_dir)],
            check=True,
        )
        tokenizer.save_pretrained(output_dir)

        if api is not None:
            repo_id = f"{args.hf_namespace}/Merged_{model_name}_{alpha_label}"
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
                    f"Upload merged debias {model_name}, alpha={alpha_label}"
                ),
            )
            print(f"Uploaded: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()