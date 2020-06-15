import json
from data_loader import dataloader
from model import classification as cls
from utils import metrics as metrics
from utils import logger
from data_loader import transform
import pandas as pd
import torch
import torch.nn as nn
import trainer
import test as tester
import logging
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import ssl
import os


def main():
    # read configure file
    with open("cfgs/tenes.cfg") as f:
        cfg = json.load(f)

    # using parsed configurations to create a dataset
    data = cfg["data"]["data_csv_name"]
    data_path = cfg["data"]["data_path"]
    batch_size = int(cfg["data"]["batch_size"])
    validation_split = float(cfg["data"]["validation_ratio"])
    # create dataset
    training_set = pd.read_csv(data, usecols=["file_name", "label"])
    train, test, _, _ = dataloader.data_split(training_set, validation_split)

    training_set = dataloader.ClassificationDataset(
        train, data_path, transform.train_transform
    )

    testing_set = dataloader.ClassificationDataset(
        test, data_path, transform.val_transform
    )
    # create dataloaders
    # global train_loader
    # global val_loader
    train_loader = torch.utils.data.DataLoader(
        training_set, batch_size=batch_size, shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        testing_set, batch_size=batch_size, shuffle=False,
    )

    print("Dataset and Dataloaders created")
    # create a model
    extractor_name = cfg["train"]["extractor"]
    model = cls.ClassificationModel(model_name=extractor_name).create_model()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)
    # convert to suitable device
    # global model
    model = model.to(device)
    print("Model created...")
    # create a metric for evaluating
    # global train_metrics
    # global val_metrics
    train_metrics = metrics.Metrics(cfg["train"]["metrics"])
    val_metrics = metrics.Metrics(cfg["train"]["metrics"])
    print("Metrics implemented successfully")

    # method to optimize the model
    # read settings from json file
    loss_function = cfg["optimizer"]["loss"]
    optimizers = cfg["optimizer"]["name"]
    learning_rate = cfg["optimizer"]["lr"]

    # initlize optimizing methods : lr, scheduler of lr, optimizer
    criterion = getattr(
        nn, loss_function, "The loss {} is not available".format(loss_function)
    )
    criterion = criterion()
    optimizer = getattr(
        torch.optim, optimizers, "The optimizer {} is not available".format(optimizers)
    )
    optimizer = optimizer(model.parameters(), lr=learning_rate)
    save_method = cfg["train"]["lr_scheduler_factor"]
    patiences = cfg["train"]["patience"]
    lr_factor = cfg["train"]["reduce_lr_factor"]
    scheduler = ReduceLROnPlateau(
        optimizer, save_method, patience=patiences, factor=lr_factor
    )

    # before training, let's create a file for logging model result
    log_file = logger.make_file(cfg["session"]["sess_name"])
    logger.log_initilize(log_file)
    print("Beginning training...")
    # training models
    num_epoch = int(cfg["train"]["num_epoch"])
    for i in range(0, num_epoch):
        loss, val_loss, train_result, val_result = trainer.train_one_epoch(
            model,
            train_loader,
            val_loader,
            device,
            optimizer,
            criterion,
            train_metrics,
            val_metrics,
        )

        # lr scheduling
        scheduler.step(val_loss)

        # export the result to log file
        logging.info("-----")
        logging.info(
            "Epoch {} / {} \n Training loss: {} - Other training metrics: ".format(
                i + 1, num_epoch, loss
            )
        )
        logging.info(train_result)
        logging.info(
            " \n Validation loss : {} - Other validation metrics:".format(val_loss)
        )
        logging.info(val_result)
        logging.info("\n")

    # testing on test set
    test_data = cfg["data"]["test_csv_name"]
    data_path = cfg["data"]["data_path"]
    test_df = pd.read_csv(test_data, usecols=["file_name", "label"])

    # prepare the dataset
    testing_set = dataloader.ClassificationDataset(
        test_df, data_path, transform.val_transform
    )

    # make dataloader
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_loader = torch.utils.data.DataLoader(testing_set, batch_size=1, shuffle=False)
    print("Inference on the testing set")
    tester.test_result(model, test_loader, device)

    # saving torch models
    torch.save(model, "saved/models/" + cfg["train"]["save_as_name"])


if __name__ == "__main__":
    main()
