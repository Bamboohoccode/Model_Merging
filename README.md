#Model Merging Implementation
This project was built for implementing the paper named An Empirical Survey of Model Merging Algorithms for Social Bias Mitigation
(Paper Link)[https://arxiv.org/pdf/2512.02689]
---
##Setup:
1. **Installing some necessary stuff**
'''bash
!git clone https://github.com/arcee-ai/mergekit.git
%cd mergekit
!pip install -e .  # install the package and make scripts available
'''
2. **Defining hyperparameters**
For Example like this
'''
MODEL_NAME = "openai-community/gpt2-medium"
WORK_DIR = "/kaggle/working/"
EPOCHS = 30
LR = 3e-5
LR_Scheduler = "linear"
3. **Finetuning model and creating the debias model**
This step finetunes the model and calculate the bias model.After having two these stuff we can calculate the debias model and upload them to HF
'''bash
cd Model_Merging
python Finetuning_model.py --name_model $MODEL_NAME \
        --work_dir $WORK_DIR \
        --epochs 1 \
        --learning_rate $LR \
        --learning_rate_scheduler $LR_Scheduler \
        --HF_TOKEN $token
'''
4. **Model Merging**
'''bash
cd Model_Merging
python merge.py --name_model $MODEL_NAME \
        --work_dir $WORK_DIR \
        --debias_model_dir $INVERSE_MODEL_DIR \
        --HF_TOKEN $token \
        --debias_model_name $INVERSE_MODEL_NAME

'''
## Result:
After step 4, 5 model with the alpha from 0 to 0.5 was uploaded to HuggingFace
May be can be updated for personal usage
