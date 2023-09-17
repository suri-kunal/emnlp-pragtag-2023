#!/usr/bin/env python
# coding: utf-8
# %%

# %%


from pathlib import Path
import pandas as pd
import numpy as np
import torch
import json
from transformers import AutoConfig, AutoTokenizer, AutoModelForMaskedLM, AutoModelWithLMHead, AutoModelForSequenceClassification, TrainingArguments, Trainer
import huggingface_hub as hf_hub
import os
from utils import to_context_free_format, CLASS_MAP, iCLASS_MAP, predictions_to_evaluation_format
import datasets
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
import random
from eval import eval_across_domains
from load import load_prediction_and_gold
import wandb


# %%


# %%
os.environ["WANDB_API_KEY"] = "get_your_own"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
hf_hub.login("get_your_own",add_to_git_credential=True)
os.environ["WANDB_PROJECT"]="emnlp_pragtag_2023"
os.environ["WANDB_MODE"]="disabled"

# %%


seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


# %%


def preprocess(item):
    item["label"] = torch.tensor(CLASS_MAP[item["label"]]).unsqueeze(0)

    return item


# %%


# tokenizer
tokenizer = "microsoft/deberta-base"
model_name = "suryakiran786/emnlp_pragtag2023_domain_adapted"
tokenizer = AutoTokenizer.from_pretrained(tokenizer,do_lower_case=True, force_download=True)

def get_model():
    model = AutoModelForSequenceClassification.from_pretrained(model_name, \
                                                               num_labels=len(CLASS_MAP), \
                                                               problem_type="single_label_classification", \
                                                               force_download=True)

    return model

# %%
def tokenize(examples):
    toks = tokenizer.batch_encode_plus(examples["txt"], padding="max_length", max_length=512, truncation=True,
                                       return_tensors="pt")
    toks["labels"] = examples["label"]

    return toks


# %%


train_file = Path.cwd().joinpath("public_data","train_inputs_full.json")

with open(train_file,"r") as f:
    train_dict_data = json.load(f)


# %%


full_data = datasets.Dataset.from_list(to_context_free_format(train_file))


# %%


full_data_df = full_data.to_pandas()


# %%


train_valid_gkf = GroupKFold()
valid_test_gkf = GroupKFold(n_splits=2)


# %%
def train_and_infer_func(train_df,valid_df,test_df):
        # Converting all dataframes to HF dataset
        train_ds = datasets.Dataset.from_pandas(train_df)

        valid_ds = datasets.Dataset.from_pandas(valid_df)
        test_ds = datasets.Dataset.from_pandas(test_df)
        data_dict = \
        datasets.DatasetDict({"train":train_ds,"valid":valid_ds,"test":test_ds})
        data_dict = \
        data_dict.map(preprocess) \
        .shuffle(seed=seed) \
        .map(tokenize, batched=True)

        # fine-tuning
        training_args = TrainingArguments(
        output_dir=f"emnlp_pragtag2023_finetuned_split_{idx}",
        overwrite_output_dir=True,
        evaluation_strategy="steps",
        logging_steps=len(train_ds) // 16,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=2 * 8,
        gradient_accumulation_steps=2,
        learning_rate=2e-5,
        weight_decay=0.01,
        adam_epsilon=1e-6,
        num_train_epochs=60,
        warmup_ratio=0.1,
        save_total_limit=4,
        save_strategy="steps",
        save_steps=len(train_ds) // 16,
        metric_for_best_model="eval_loss",
        load_best_model_at_end=True,
        greater_is_better=False,
        )

        trainer = Trainer(
        model_init=get_model,
        args=training_args,
        train_dataset=data_dict["train"],
        eval_dataset=data_dict["valid"],
        # data_collator=data_collator,
        )

        trainer.train()

        predictions = trainer.predict(data_dict["test"])
        predicted_classes = np.argmax(predictions.predictions,axis=-1)

        prediction_list = []
        for test_datapoint, prediction in zip(data_dict["test"],predicted_classes):
            prediction_list.append({"sid": test_datapoint["sid"], "label": iCLASS_MAP[prediction]})

        r = predictions_to_evaluation_format(prediction_list)

        pred_path = f"predicted_split_{idx}.json"
        with open(pred_path, "w+") as f:
            json.dump(r, f, indent=4)

        id_list = []
        for data in r:
            id_list.append(data["id"])

        test_dict_data = []
        for elem in train_dict_data:
            if elem["id"] in id_list:
                test_dict_data.append(elem)

        gold_path = f"gold_data_split_{idx}.json"
        with open(gold_path,"w") as f:
            json.dump(test_dict_data,f,indent=4)

        pred, gold = load_prediction_and_gold(pred_path, gold_path)
        per_domain, mean = eval_across_domains(gold, pred)

        out_path = f"scores_split_{idx}.txt"
        with open(out_path, "w+") as f:
            for k, v in per_domain.items():
                f.write(f"f1_{k}:{v}\n")
            f.write(f"f1_mean:{mean}")

        wandb.finish()


# %%
if __name__ == "__main__":

    for idx,(train_idx, valid_idx) in enumerate(train_valid_gkf.split(X=full_data_df,y=full_data_df["label"],groups=full_data_df["report_id"])):
        # Splitting data into Train and validation
        train_df = full_data_df.loc[train_idx,:]    
        og_valid_df = full_data_df.loc[valid_idx,:]

        # Splitting validation data into validation and test
        valid_idx,test_idx = \
        next(iter(valid_test_gkf.split(X=og_valid_df,y=og_valid_df["label"],groups=og_valid_df["report_id"])))
        valid_df = og_valid_df.iloc[valid_idx]
        test_df = og_valid_df.iloc[test_idx]
        train_and_infer_func(train_df,valid_df,test_df)
