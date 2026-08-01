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

def create_inverse_state(
    base_state: dict[str, torch.Tensor],
    bias_state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    base_keys = set(base_state)
    bias_keys = set(bias_state)

    missing = sorted(base_keys - bias_keys)
    unexpected = sorted(bias_keys - base_keys)
    if missing or unexpected:
        raise ValueError(
            "The checkpoint is incompatible with the base model.\n"
            f"Missing keys (first 10): {missing[:10]}\n"
            f"Unexpected keys (first 10): {unexpected[:10]}"
        )

    inverse_state: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for name, base_tensor in base_state.items():
            bias_tensor = bias_state[name]
            if base_tensor.shape != bias_tensor.shape:
                raise ValueError(
                    f"Shape mismatch at {name}: base={tuple(base_tensor.shape)}, "
                    f"bias={tuple(bias_tensor.shape)}"
                )

            base_cpu = base_tensor.detach().cpu()
            bias_cpu = bias_tensor.detach().cpu()
            if torch.is_floating_point(base_cpu):
                inverse_state[name] = (
                    2.0 * base_cpu.float() - bias_cpu.float()
                )
            else:
                inverse_state[name] = base_cpu.clone()

    return inverse_state


def verify_saved_model(model_dir: Path, tokenizer_length: int) -> None:
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    input_shape = tuple(model.get_input_embeddings().weight.shape)
    output_shape = tuple(model.get_output_embeddings().weight.shape)

    if input_shape[0] != tokenizer_length:
        raise ValueError(
            f"Invalid vocabulary after merge: embedding rows={input_shape[0]}, "
            f"tokenizer length={tokenizer_length}"
        )

    print(
        f"Verified {model_dir.name}: input={input_shape}, "
        f"output={output_shape}, tokenizer={tokenizer_length}"
    )
    del model
    gc.collect()


DEFAULT_BASE_MODEL = "ComCom/gpt2-small"
DEFAULT_BIAS_PTH = (
    "/kaggle/input/notebooks/mrrobotbamboo/"
    "finetuning-for-model-merging-ipynb/GPT2_Small.pth"
)
DEFAULT_WORK_DIR = "/kaggle/working"
DEFAULT_HF_NAMESPACE = "trinhkhng"
DEFAULT_ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
#----------------------------------------------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create inverse weights and merge them with the base model."
    )
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--bias-pth", default=DEFAULT_BIAS_PTH)
    parser.add_argument("--work-dir", default=DEFAULT_WORK_DIR)
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
    bias_path = Path(args.bias_pth)

    if not bias_path.is_file():
        raise FileNotFoundError(f"Bias checkpoint not found: {bias_path}")

    model_name = args.base_model.split("/")[-1]

    base_model_dir = work_dir / "Base_Model"
    inverse_model_dir = work_dir / "Inverse_Model"
    config_path = work_dir / "merge_config.yml"
    # Load base model
    print(f"Loading base model: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        use_fast=False,
    )

    checkpoint = torch.load(
        bias_path,
        map_location="cpu",
        weights_only=True,
    )
    bias_state = unwrap_state_dict(checkpoint)
    inverse_state = create_inverse_state(base_model.state_dict(), bias_state)

    base_model.save_pretrained(base_model_dir, safe_serialization=True)
    tokenizer.save_pretrained(base_model_dir)
    # load inverse model
    inverse_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    inverse_model.load_state_dict(inverse_state, strict=True)
    inverse_model.save_pretrained(inverse_model_dir, safe_serialization=True)
    tokenizer.save_pretrained(inverse_model_dir)

    # del checkpoint, bias_state, inverse_state, base_model, inverse_model
    # gc.collect()

    token = None if args.skip_upload else os.environ.get("HF_TOKEN")
    if not args.skip_upload and not token:
        raise RuntimeError(
            "HF_TOKEN is not set. Set a Hugging Face write token or use "
            "--skip-upload."
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
        verify_saved_model(output_dir, len(tokenizer))

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