# 
import os
import numpy as np
import pandas as pd
import torch 
from transformers import BertTokenizer, BertForSequenceClassification


class Dataset(torch.utils.data.Dataset):
    
    def __init__(
        self,
        root,
        max_length=512,
    ):
        
        self.root = root
        self.max_length = max_length

        # load moive review data
        data = pd.read_csv(os.path.join(root, "movie.csv")).values
        reviews, labels = data[:, 0], data[:, 1].astype(np.int64)

        # get 1K positive instances and 1K negative instances
        n_instances = 1000 
        index = np.where(labels == 0)[0].tolist()[:n_instances] + np.where(labels == 1)[0].tolist()[:n_instances]
        reviews, labels = reviews[index], labels[index]

        # load bert tokenizer
        tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased",
            do_lower_case=True,
        )
        
        # tokenize
        input_ids, attention_mask = [], []
        for review in reviews:
            
            encoding = tokenizer(
                review,    
                add_special_tokens=True,
                max_length=self.max_length,
                # pad_to_max_length=True,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
    
            input_ids.append(encoding["input_ids"])
            attention_mask.append(encoding["attention_mask"])
    
        self.input_ids = torch.cat(input_ids, dim=0)
        self.attention_mask = torch.cat(attention_mask, dim=0)
        self.labels = torch.tensor(labels, dtype=torch.int64) 

    def __len__(self):
        # number of data in the dataset
        return len(self.input_ids)

    def __getitem__(self, idx):
        # return one instance 
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }

def load_model(
    n_classes,
    n_encoder_layers,
):

    # load bert model for sequence classification
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=n_classes,
    )

    # freeze embedding layers and encoder layers for fine tuning
    for params in model.bert.embeddings.parameters():
        params.requires_grad = False
    for params in model.bert.encoder.layer[:n_encoder_layers].parameters():
        params.requires_grad = False
    
    return model

