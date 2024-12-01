# import modules 
import time
import numpy as np

import torch

import utils

def train():
    '''
    Pipeline for training BLIP model
    '''

    # --- load device --- # 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # --- load dataset --- #
    path = "../../../data"    
    batch_size = 4

    # load dataset
    dataset = Dataset(
        root=path,
        input_max_length=64,
        output_max_length=10,
        train=True,
        seed=0,
    )

    # train/valid/test subsets
    torch.manual_seed(0) # set torch random seed
    indices = torch.randperm(len(dataset)).tolist()

    train_split = 0.8
    train_index = int(len(indices) * train_split)

    dataset_train = torch.utils.data.Subset(dataset, indices[:train_index]) 
    dataset_valid = torch.utils.data.Subset(dataset, indices[train_index:]) 

    # define train/valid/test data loaders
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
    )

    data_loader_valid = torch.utils.data.DataLoader(
        dataset_valid,
        batch_size=batch_size,
        shuffle=False,
    )


    # --- load model --- #
    model = utils.load_model().to(device)
   
    for name, params in model.named_parameters():
        print(name, params.shape, params.requires_grad)
    

    # --- load optimizer --- #
    optim = torch.optim.Adam(
        model.parameters(),
        lr=3e-5,
    )

    # --- load learning rate scheduler --- #
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optim,
        mode="min",
        min_lr=1e-6,
    )

    # --- train model --- #
    n_epochs = 2
    best_valid_loss = float("inf") # best validation loss 

    metric = {
        "train_loss": [],
        "valid_loss": [],
    }

    for epoch in range(n_epochs):
       
        # train 
        train_loss = 0
        model.train()
        for i, batch in enumerate(data_loader_train):

            # move data to cuda
            batch = {key: val.to(device) for key, val in batch.items()}

            # model output 
            outputs = model(
                pixel_values=batch["pixel_values"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                decoder_input_ids=batch["decoder_input_ids"],
                decoder_attention_mask=batch["decoder_attention_mask"],
                labels=batch["labels"],
            )

            # backpropagation
            model.zero_grad()
            outputs.loss.backward()
            optim.step()
        
            # collect train loss
            train_loss += outputs.loss.item()

        train_loss /= len(data_loader_train)


        # validation
        valid_loss = 0
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(data_loader_valid):

                # move data to cuda 
                batch = {key: val.to(device) for key, val in batch.items()}

                # model output 
                outputs = model(
                    pixel_values=batch["pixel_values"],
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    decoder_input_ids=batch["decoder_input_ids"],
                    decoder_attention_mask=batch["decoder_attention_mask"],
                    labels=batch["labels"],
                )

                # calculate validation loss
                valid_loss += outputs.loss.item()

            valid_loss /= len(data_loader_valid)

        # update saved model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss

            # save model weight
            torch.save(model.text_decoder.cls.state_dict(), "decoder_cls.pt")

        # save metrics
        metric["train_loss"].append(train_loss)
        metric["valid_loss"].append(valid_loss)

        # update the learning rate
        lr_scheduler.step(valid_loss)

        print(train_loss, valid_loss)

    np.save("train_valid_metric.npy", metric)


def main():
    # implement train function
    train()

if __name__ == "__main__":

    time_1 = time.time()
    main()
    time_2 = time.time()

    print(time_2 - time_1)
