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
    dataset_test = utils.Dataset(
        root=path,
        input_max_length=64,
        output_max_length=6,
        train=False,
        seed=0,
    )


    # --- load model --- #
    model = load_model().to(device)
    # load model weight
    model_weight_path = "decoder_cls.pt"
    model.text_decoder.cls.load_state_dict(torch.load(model_weight_path, weights_only=True))

    model.eval()


    # --- generate answers --- #

    # beam search 
    beam_search_config = {
        "do_sample": False,
        "num_beams": 50,
        "num_return_sequences": 10,
    }
    
    result = {
        "target": [],
        "generation": [],
    }

    for i, instance in enumerate(dataset_test):
        
        # move data to cuda
        instance = {key: val.unsqueeze(0).to(device) for key, val in instance.items()}

        # generate answer
        output_ids = model.generate(
            pixel_values=instance["pixel_values"],
            input_ids=instance["input_ids"],
            attention_mask=instance["attention_mask"],
            max_new_tokens=100,
            **beam_search_config,
        )

        # decodes
        temp = []
        for curr_output_ids in output_ids:
            output = dataset_test.processor.decode(
                curr_output_ids,
                skip_special_tokens=True,
            )
            temp.append(output)

        # save result
        result["generation"].append(temp)

    result["target"] = dataset_test.dataset
    
    # save results
    np.save("answers.npy", result)


def main():
    # implement generate function
    generate()

if __name__ == "__main__":

    time_1 = time.time()
    main()
    time_2 = time.time()

    print(time_2 - time_1)
