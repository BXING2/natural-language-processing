# import modules 
import time
import numpy as np

import torch

import utils

def generate():
    '''
    Pipeline for generating sentences using the trained model.
    '''

    # --- load device --- # 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0) # set torch random seed

    # --- load dataset --- #
    path = "../../data"    
    batch_size = 32

    dataset = utils.Dataset(
        root=path,
        max_length=128,
    )

    # --- load model --- #
    n_tokens = len(dataset.tokenizer)
    n_decoder_layers = 10
    model = utils.load_model(
        n_tokens=n_tokens,
        n_decoder_layers=n_decoder_layers,
    ).to(device)
   
    for name, params in model.named_parameters():
        print(name, params.shape, params.requires_grad)
    
    # load model weights
    model_weight_path = "../model_2/"
    model.transformer.wte.load_state_dict(torch.load(model_weight_path + "token_embedding.pt", weights_only=True))
    model.transformer.h[n_decoder_layers:].load_state_dict(torch.load(model_weight_path + "decoder_layer.pt", weights_only=True))
    model.transformer.ln_f.load_state_dict(torch.load(model_weight_path + "layer_norm.pt", weights_only=True))

    model.eval()

    # make eos/pad token in the generation config and tokenizer consistent 
    model.generation_config.pad_token_id = dataset.tokenizer.pad_token_id
    model.generation_config.eos_token_id = dataset.tokenizer.eos_token_id


    # --- generate sentence --- #
    # set up prompt words (here, only the start of sentence token is provided)
    prompt = "<|startoftext|>"

    # encode input
    encoding = dataset.tokenizer(
        prompt,                      
        return_attention_mask=True,
        return_tensors="pt",
    )
   
    encoding = {key: val.to(device) for key, val in encoding.items()}

    # configs of different sampling methods
    # greedy decoding
    greedy_config = {
        "do_sample": False,
        "num_beams": 1,
    }
    # beam search 
    beam_search_config = {
        "do_sample": False,
        "num_beams": 10,
        "repetition_penalty": 2.0,
    }
    # top p (nucleus)
    top_p_config = {
        "do_sample": True,
        "top_p": 0.6,
        #"temperature": 5., 
    }
    # top k 
    top_k_config = {
        "do_sample": True,
        "top_k": 20,
        "temperature": 1.,
    }

    # generate sentences
    output_ids = model.generate(
        input_ids=encoding["input_ids"],
        attention_mask=encoding["attention_mask"],
        max_length=100,
        **greedy_config,
        #**beam_search_config,
        #**top_p_config,
        #**top_k_config,
    )

    # decodes
    f = open("outputs.txt", "w")
    for cur_output_ids in output_ids:
        output = dataset.tokenizer.decode(
            cur_output_ids,
            skip_special_tokens=True,
        )

        print(output)
        f.write(output + "\n")
    f.close()


def main():
    # implement generate function
    generate()

if __name__ == "__main__":

    time_1 = time.time()
    main()
    time_2 = time.time()

    print(time_2 - time_1)
