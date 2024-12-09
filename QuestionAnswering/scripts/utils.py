# 
import os
import numpy as np
import pandas as pd

import torch 
from transformers import AutoTokenizer, AutoModelForCausalLM

# others 

class Dataset(torch.utils.data.Dataset):
    # load dataset for training question answering system 
    def __init__(
        self,
        root, # root directory of data
        max_length=512, # max sequence length 
    ):
        
        self.root = root
        self.max_length = max_length

        # load data, Nx3: question, answer, context
        file_path = os.path.join(root, "Financial-QA-10k.csv")
        dataset = pd.read_csv(
            filepath_or_buffer=file_path,
            sep=",",
            header=0,
            index_col=False,
            usecols=[0,1,2],
        )
            
        # drop rows containing any missing values 
        dataset.dropna(
            axis=0, 
            how="any",
            inplace=True,
        )
        
        # select 1K instances to finetune LLaMA model 
        dataset = dataset.values[:1000] 

        # load llama tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.2-1B",
            token=<TOKEN>, # authorized token needed
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
            pad_token="<pad>",
            add_bos_token=False,
        )
       
        # add new prompt words 
        self.tokenizer.add_tokens(["<Context>", "<Question>", "<Answer>"])
        self.dataset = []
        
        # tokenize
        input_ids, attention_mask, labels = [], [], []
        for i in range(len(dataset)):
        
            # get question, answer and context
            question, answer, context = dataset[i]
            
            # tokenize prompt
            context_question = "<Context>" + context + "<Question>" + question + "<Answer>"
            context_question_tokens = self.tokenizer.tokenize(context_question)
            context_question_token_ids = self.tokenizer.convert_tokens_to_ids(context_question_tokens)

            # tokenize answer with eos_token
            answer_tokens = self.tokenizer.tokenize(answer + self.tokenizer.eos_token)
            answer_token_ids = self.tokenizer.convert_tokens_to_ids(answer_tokens)
           
            # if the context_question is too long, do not consider it. 
            if len(context_question_tokens) >= self.max_length:
                continue

            curr_input_ids = context_question_token_ids + answer_token_ids # all token ids of an instance
            curr_length = len(curr_input_ids) 
            if curr_length < self.max_length: # add pad tokens
                curr_input_ids += [self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)] * (self.max_length - curr_length)
                curr_attention_mask = [1] * curr_length + [0] * (self.max_length - curr_length)
            else: # truncation 
                curr_input_ids = curr_input_ids[: self.max_length]
                curr_attention_mask = [1] * self.max_length 

            # mask context and question for not computing loss 
            curr_labels = curr_input_ids.copy() # labels are same with input for self-supervised learning
            for index in range(len(context_question_tokens)):
                curr_labels[index] = -100

            # store input token ids, label ids and attention_masks
            input_ids.append(curr_input_ids)
            attention_mask.append(curr_attention_mask)
            labels.append(curr_labels)
       
            self.dataset.append(dataset[i])
 
        # convert input_ids, attention_mask, labels to tensor 
        self.input_ids = torch.tensor(input_ids)
        self.attention_mask = torch.tensor(attention_mask)
        self.labels = torch.tensor(labels)

        # set ids of padding tokens as -100
        self.labels[~self.attention_mask.bool()] = -100

        # print(len(self.dataset), self.input_ids.shape, self.labels.shape, self.attention_mask.shape)
        

    def __len__(self):
         
        return len(self.input_ids)

    def __getitem__(self, idx):

        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


def load_model(
    n_tokens, # number of tokens in the vocabulary
    n_decoder_layers, # number of decoder layers to freeze
):

    # load llama model
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B",
        token=<TOKEN>, # authorized token needed
    )

    # resize embedding dimension due to added special tokens
    model.resize_token_embeddings(n_tokens)

    # freeze decoder layers
    for params in model.model.layers[:n_decoder_layers].parameters():
        params.requires_grad = False
    
    return model

