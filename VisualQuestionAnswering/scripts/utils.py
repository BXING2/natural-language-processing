# 
import os
from PIL import Image
import numpy as np
import pandas as pd

import torch 
from transformers import (
    AutoProcessor,
    BlipProcessor,
    BlipForQuestionAnswering,
    DataCollatorWithPadding,
)

class Dataset(torch.utils.data.Dataset):
    '''
    module for loading dataset
    '''
    def __init__(
        self,
        root, # root directory of data
        input_max_length=512, # max sequence length of input
        output_max_length=512, # max sequence length of output 
        train=False, # training or inference
        seed=0, # numpy random seed
    ):
        
        self.root = root
        self.input_max_length = input_max_length
        self.output_max_length = output_max_length

        # load dataset
        if train:  # train data
            self.dataset = pd.read_csv(
                os.path.join(
                    root,
                    "data_train.csv",
                ),
            ).values

        else: # test data
            self.dataset = pd.read_csv(
                os.path.join(
                    root,
                    "data_eval.csv",
                ),
            ).values

        # load dataset subset
        np.random.seed(seed)
        permu_index = np.random.permutation(range(len(self.dataset)))[:1000] # take 1000 instances
        self.dataset = self.dataset[permu_index]

        # replace "_" with " "
        for i in range(len(self.dataset)):
            self.dataset[i, 1] = self.dataset[i, 1].replace("_", " ")

        # load processor
        self.processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-vqa-base",
        )

        # load data collator
        self.data_collator = DataCollatorWithPadding(
            tokenizer=self.processor.tokenizer,
        )
    
    def __len__(self):

        return len(self.dataset)

    def __getitem__(self, idx):
        
        # load question, answer and image 
        question, answer, image_id = self.dataset[idx]
        image = Image.open(
            os.path.join(
                self.root,
                "images",
                "{}.png".format(image_id),
            ),
        )
    
        # encode image and question 
        encoding = self.processor(
            images=image,
            text=question,
            max_length=self.input_max_length,
            padding="max_length",
            truncation=True, 
            return_tensors="pt",
        ) 

        # encode answer
        labels = self.processor(
            text=answer,
            max_length=self.output_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        # decoder input ids and attention mask
        encoding["decoder_input_ids"] = labels.input_ids
        encoding["decoder_attention_mask"] = labels.attention_mask

        # labels 
        encoding["labels"] = labels.input_ids.detach().clone()
        encoding["labels"][~labels.attention_mask.bool()] = -100
        
        encoding = {key: val.squeeze() for key, val in encoding.items()}
         
        #for key, val in encoding.items():
        #    print(key, val.shape)
        return encoding

def load_model():

    # load pre-trained blip model
    model = BlipForQuestionAnswering.from_pretrained(
        "Salesforce/blip-vqa-base",
    )
    
    # freeze model weights 
    for params in model.parameters():
        params.requires_grad = False
    for params in model.text_decoder.cls.parameters(): 
        params.requires_grad = True   

    return model

