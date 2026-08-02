# Model Merging Implementation

This project implements the methods described in the paper  
[An Empirical Survey of Model Merging Algorithms for Social Bias Mitigation](https://arxiv.org/pdf/2512.02689).

---

## Setup

### 1. Install dependencies

```bash
git clone https://github.com/arcee-ai/mergekit.git
cd mergekit
pip install -e .
```

> In Kaggle notebooks, you may use `!git clone`, `%cd`, and `!pip install` instead.

### 2. Define hyperparameters

For example:

```python
MODEL_NAME = "openai-community/gpt2-medium"
WORK_DIR = "/kaggle/working"
EPOCHS = 30
LR = 3e-5
LR_SCHEDULER = "linear"
```

### 3. Fine-tune the model and create the debias model

This step fine-tunes the base model, calculates the bias model, then creates a debias model.

```bash
cd Model_Merging

python Finetuning_model.py \
  --name_model "$MODEL_NAME" \
  --work_dir "$WORK_DIR" \
  --epochs "$EPOCHS" \
  --learning_rate "$LR" \
  --learning_rate_scheduler "$LR_SCHEDULER" \
  --HF_TOKEN "$HF_TOKEN"
```

### 4. Merge models

```bash
cd Model_Merging

python merge.py \
  --name_model "$MODEL_NAME" \
  --work_dir "$WORK_DIR" \
  --debias_model_dir "$INVERSE_MODEL_DIR" \
  --debias_model_name "$INVERSE_MODEL_NAME" \
  --HF_TOKEN "$HF_TOKEN"
```

## Results

After the merging step, five merged models using alpha values from `0.0` to `0.5` are uploaded to Hugging Face.

## Notes

- Set `HF_TOKEN` before uploading models:

  ```bash
  export HF_TOKEN="your_huggingface_write_token"
  ```

- Never commit your Hugging Face token to GitHub.
- This repository may be extended for personal experiments with different base models, datasets, merging methods, and alpha values.
