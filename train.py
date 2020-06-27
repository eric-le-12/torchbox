import json
from data_loader import dataloader
from model import classification as cls
from utils import metrics as metrics
from utils import logger
from utils import custom_loss
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
from datetime import datetime

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
    try:
        # if the loss function comes from nn package
        criterion = getattr(
        nn, loss_function, "The loss {} is not available".format(loss_function)
    )
    except:
        # use custom loss
        criterion = getattr(
        custom_loss, loss_function, "The loss {} is not available".format(loss_function)
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
    time_str = str(datetime.now().strftime("%Y%m%d-%H%M"))
    log_file = logger.make_file(cfg["session"]["sess_name"],time_str)
    logger.log_initilize(log_file)
    print("Beginning training...")
    # export the result to log file
    f = open("cfgs/tenes.cfg","w+")
    logging.info("-----")
    logging.info("session name: {} \n".format(cfg["session"]["sess_name"]))
    logging.info(model)
    logging.info("\n")
    logging.info("CONFIGS")
    # logging the configs:
    for line in f:
        logging.info(line)
        logging.info("\n")
    # training models
    num_epoch = int(cfg["train"]["num_epoch"])
    best_val_acc = 0
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
        # saving epoch with best validation accuracy
        if (best_val_acc < float (val_result["accuracy_score"])):
            print("Validation accuracy= ",val_result["accuracy_score"], "===> Save best epoch")
            best_val_acc = val_result["accuracy_score"]
            torch.save(model.state_dict(), "saved/models/" +  time_str+'-'+cfg["train"]["save_as_name"])
        else:
            print("Validation accuracy= ",val_result["accuracy_score"], "===> No saving")
            continue

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
    
    # load the test model and making inference
    test_model = cls.ClassificationModel(model_name=extractor_name).create_model()
    model_path = os.path.join("saved/models", time_str+'-'+cfg["train"]["save_as_name"])
    test_model.load_state_dict(torch.load(model_path))
    test_model = test_model.to(device)
    logging.info(tester.test_result(test_model, test_loader, device))

    # saving torch models


if __name__ == "__main__":
    main()
