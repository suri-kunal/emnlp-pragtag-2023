#!/usr/bin/env python
# coding: utf-8
# %%

# %%


from pathlib import Path
import torch
import json
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
import huggingface_hub as hf_hub
import os
from utils import (
    to_context_free_format,
    CLASS_MAP,
    iCLASS_MAP,
    predictions_to_evaluation_format,
)
import datasets
from torch.utils.data import DataLoader
from tqdm import tqdm

# %%


# %%
os.environ["WANDB_API_KEY"] = "get_your_own"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
hf_hub.login("get_your_own", add_to_git_credential=True)
os.environ["WANDB_PROJECT"] = "emnlp_pragtag_2023"

# %%

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def preprocess(item):
    if "label" in item:
        item["label"] = torch.tensor(CLASS_MAP[item["label"]]).unsqueeze(0)

    return item


# %%


# tokenizer
tokenizer = "microsoft/deberta-base"
tokenizer = AutoTokenizer.from_pretrained(
    tokenizer, do_lower_case=True, force_download=True
)

model_dict = {}
for split in range(5):
    model_name = f"suryakiran786/emnlp_pragtag2023_finetuned_test_split_{split}"
    model = AutoModelForSequenceClassification.from_pretrained(model_name,
                                                               num_labels=len(CLASS_MAP),
                                                               problem_type="single_label_classification",
                                                               force_download=True
                                                              )
    model_dict[split] = model


def get_model():
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(CLASS_MAP),
        problem_type="single_label_classification",
        force_download=True,
    )

    return model


# %%
def tokenize(examples):
    toks = tokenizer.batch_encode_plus(
        examples["txt"],
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt",
    )
    if "label" in examples:
        toks["labels"] = examples["label"]

    return toks

def create_preds(split,batch_input):
#    model_name = f"suryakiran786/emnlp_pragtag2023_finetuned_test_split_{split}"
#    model = AutoModelForSequenceClassification.from_pretrained(model_name,
#                                                               num_labels=len(CLASS_MAP),
#                                                               problem_type="single_label_classification",
#                                                               force_download=True
#                                                              )

    model = model_dict[split]

    model.eval()
    model.to(device)

    batch_outputs = model(
            input_ids=torch.stack(batch_inputs["input_ids"])
            .transpose(1, 0)
            .to(device),
            attention_mask=torch.stack(batch_inputs["attention_mask"])
            .transpose(1, 0)
            .to(device)
            )

    return batch_outputs

# %%
if __name__ == "__main__":
    train_file = Path.cwd().joinpath("public_secret", "test_inputs.json")

    eval_data = (
        datasets.Dataset.from_list(to_context_free_format(train_file))
        .map(preprocess)
        .map(tokenize, batched=True)
    )

    batch_size = 3
    inputs = DataLoader(eval_data, batch_size=batch_size)

    predictions = []

    for batch_inputs in tqdm(inputs, desc="Iterating over input"):
        list_of_predictions = []

        for split in range(5):            

            try:
                batch_outputs = create_preds(split,batch_inputs)
            
                list_of_predictions.append(batch_outputs.logits[None,:])
            except Exception as e:
                print(e)

        final_predictions = torch.cat(list_of_predictions, dim=0)

        final_predictions = torch.mean(final_predictions,dim=0)

        final_predictions = torch.argmax(final_predictions, dim=-1)

        num_of_iterations = min(batch_size, len(final_predictions))
        print(num_of_iterations)
        
        predictions += [
                    {
                        "sid": batch_inputs["sid"][i],
                        "label": iCLASS_MAP[final_predictions[i].item()],
                    }
                    for i in range(num_of_iterations)
                ]
        print(len(predictions))

    r = predictions_to_evaluation_format(predictions)
    print(len(r))

    with open("predicted.json", "w+") as f:
        json.dump(r, f, indent=4)
