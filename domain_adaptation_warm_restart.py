#!/usr/bin/env python
# coding: utf-8
# %%

# %%

import json
import os
from pathlib import Path

import huggingface_hub as hf_hub
import pandas as pd
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
)
import os
# %%


# %%
os.environ["WANDB_API_KEY"] = "23e6940ba17fe0fd2bf2616685c3978f2ce87d7b"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
hf_hub.login("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", add_to_git_credential=True)
os.environ["WANDB_PROJECT"] = "emnlp_pragtag_2023"


# %%


non_empty_review_list = []
for r in (
    Path.cwd().joinpath("auxilliary_data", "F1000-22", "data").glob("**/reviews.json")
):
    with open(r, "r") as f:
        review = json.load(f)
    if len(review) > 0:
        non_empty_review_list.append(r)


# %%


review_id_list = []
review_text_list = []
for ner in non_empty_review_list:
    with open(ner, "r") as f:
        review_list = json.load(f)
    for review in review_list:
        review_id_list.append(review["rid"])
        review_text_list.append(review["report"]["main"])


# %%


abstract_data = pd.DataFrame.from_dict(
    data={"review_id": review_id_list, "review_text": review_text_list}
)


# %%


from sklearn.model_selection import train_test_split

# %%


train_abstract_data, test_abstract_data = train_test_split(
    abstract_data, test_size=0.5, random_state=42
)
valid_abstract_data, test_abstract_data = train_test_split(
    test_abstract_data, test_size=0.5, random_state=42
)


# %%


import datasets

# %%


train_dataset = datasets.Dataset.from_pandas(train_abstract_data)
valid_dataset = datasets.Dataset.from_pandas(valid_abstract_data)
test_dataset = datasets.Dataset.from_pandas(test_abstract_data)


# %%


abstract_hf_dataset = datasets.DatasetDict(
    {"train": train_dataset, "valid": valid_dataset, "test": test_dataset}
)

# %%

tokenizer = "microsoft/deberta-base"
model_name = "./emnlp_pragtag2023_domain_adapted"
tokenizer = AutoTokenizer.from_pretrained(
    tokenizer, do_lower_case=True, force_download=True
)


# %%


def preprocess_text(example):
    return tokenizer(example["review_text"])


# %%


abstract_hf_dataset_tokenised = abstract_hf_dataset.map(
    preprocess_text,
    batched=True,
    remove_columns=abstract_hf_dataset["train"].features,
    num_proc=10,
)


# %%


block_size = 512


def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {}
    for k in examples.keys():
        tmp = sum(examples[k], [])
        concatenated_examples[k] = tmp
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder
    # we could add padding if the model supported it instead of this drop
    # you can customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of block_size.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    return result


# %%


abstract_hf_dataset_tokenised_chunked = abstract_hf_dataset_tokenised.map(
    group_texts, batched=True, num_proc=1
)


# %%


from transformers import DataCollatorForLanguageModeling

tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm_probability=0.15
)


# %%


model = AutoModelForMaskedLM.from_pretrained(model_name)


# %%


from transformers import Trainer, TrainingArguments

# %%


batch_size = 8
gradient_accumulation_steps = 2
num_epochs = 100
training_args = TrainingArguments(
    output_dir="emnlp_pragtag2023_domain_adapted_warm_restart",
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=2 * batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    learning_rate=2e-5,
    weight_decay=0.01,
    adam_epsilon=1e-6,
    num_train_epochs=num_epochs,
    warmup_ratio=0.1,
    save_total_limit=4,
    push_to_hub=True,
    save_strategy="epoch",
    run_name=model_name.split("/")[-1],
    metric_for_best_model="eval_loss",
    load_best_model_at_end=True,
    greater_is_better=False,
    report_to="wandb",
    hub_strategy="end",
    hub_private_repo=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=abstract_hf_dataset_tokenised_chunked["train"],
    eval_dataset=abstract_hf_dataset_tokenised_chunked["valid"],
    data_collator=data_collator,
)

trainer.train()
# %%
