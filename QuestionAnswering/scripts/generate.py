# import modules 
import time
import numpy as np

import torch

import utils

def generate():
    '''
    Pipeline for generating answers for given contexts and questions using the trained model.
    '''

    # --- load device --- # 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0) # set torch random seed

    # --- load dataset --- #
    path = "../../data"    
    dataset = utils.Dataset(
        root=path,
        max_length=256,
    )

    # train/valid/test split
    indices = torch.randperm(len(dataset)).tolist()

    train_split, valid_split, test_split = 0.6, 0.2, 0.2
    train_index = int(len(indices) * train_split)
    valid_index = int(len(indices) * valid_split) + train_index

    # test dataset 
    dataset_test = dataset.dataset[indices[valid_index:]]


    # --- load model --- #
    n_tokens = len(dataset.tokenizer)
    n_decoder_layers = 15
    model = utils.load_model(
        n_tokens=n_tokens,
        n_decoder_layers=n_decoder_layers,
    ).to(device)

    for name, params in model.named_parameters():
        print(name, params.shape, params.requires_grad)

    # load model weights
    model_weight_path = "5e-6"
    embed_pt_file, layer_pt_file, norm_pt_file = "embed.pt", "layers.pt", "rms_norm.pt"
    model.model.embed_tokens.load_state_dict(torch.load(os.path.join(model_weight_path, embed_pt_file), weights_only=True))
    model.model.layers[num_decoder_layers:].load_state_dict(torch.load(os.path.join(model_weight_path, layer_pt_file), weights_only=True))
    model.model.norm.load_state_dict(torch.load(os.path.join(model_weight_path, norm_pt_file), weights_only=True))

    model.eval()

    # make eos/pad token in the generation config and tokenizer consistent 
    
    model.generation_config.bos_token_id = dataset.tokenizer.bos_token_id
    model.generation_config.eos_token_id = dataset.tokenizer.eos_token_id
    model.generation_config.pad_token_id = dataset.tokenizer.pad_token_id
    model.generation_config.unk_token_id = dataset.tokenizer.unk_token_id


    # --- generate answers --- #

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
    
    result = {
        "target": [],
        "generation": [],
    }

    for i, instance in enumerate(dataset_test):

        # construct inputs from question and context
        question, answer, context = instance
        inputs = "<Context>" + context + "<Question>" + question + "<Answer>"

        # encode inputs
        encoding = dataset.tokenizer(
            inputs,                      
            return_attention_mask=True,  
            return_tensors="pt",
        )

        encoding = {key: val.to(device) for key, val in encoding.items()}

        # generate answer
        output_ids = model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_new_tokens=100,
            num_return_sequences=1,
        )

        # decode output ids 
        temp = []
        for curr_output_ids in output_ids:
            output = dataset.tokenizer.decode(
                curr_output_ids[len(input_ids[0]):],
                skip_special_tokens=True,
            )
            temp.append(output)

        # save result
        result["target"].append(answer)
        result["generation"].append(temp)
    
    np.save("answers.npy", result)


def main():
    # implement generate function
    generate()

if __name__ == "__main__":

    time_1 = time.time()
    main()
    time_2 = time.time()

    print(time_2 - time_1)
