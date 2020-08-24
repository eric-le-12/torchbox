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
from torch.optim.lr_scheduler import CosineAnnealingLR
import ssl
import os
from datetime import datetime
import argparse
from torchsampler import ImbalancedDatasetSampler


def main():
    parser = argparse.ArgumentParser(description='NA')
    parser.add_argument('-c', '--configure', default='cfgs/chexphoto.cfg', help='JSON file')
    parser.add_argument('-cp', '--checkpoint', default=None, help = 'checkpoint path')
    args = parser.parse_args()
    checkpoint = args.checkpoint
    # read configure file
    with open(args.configure) as f:
        cfg = json.load(f)
    time_str = str(datetime.now().strftime("%Y%m%d-%H%M"))
    tensorboard_writer  = logger.make_writer(cfg["session"]["sess_name"], time_str) 
    # using parsed configurations to create a dataset
    data = cfg["data"]["data_csv_name"]
    valid = cfg['data']['test_csv_name']
    data_path = cfg["data"]["data_path"]
    batch_size = int(cfg["data"]["batch_size"])
    validation_split = float(cfg["data"]["validation_ratio"])
    # create dataset
    training_set = pd.read_csv(data, usecols=["file_name", "label"])
    valid_set = pd.read_csv(valid, usecols=["file_name", "label"])
    # train, test, _, _ = dataloader.data_split(training_set, validation_split)

    training_set = dataloader.ClassificationDataset(
        training_set, data_path, transform.train_transform
    )

    testing_set = dataloader.ClassificationDataset(
        valid_set, data_path, transform.val_transform
    )
    # create dataloaders
    # global train_loader
    # global val_loader
    #SAmpler to prevent inbalance data label
    # train_loader = torch.utils.data.DataLoader(training_set,sampler=ImbalancedDatasetSampler(training_set, callback_get_label=lambda x, i: tuple(x[i][1].tolist())),batch_size=batch_size,)

    #End sampler
    train_loader = torch.utils.data.DataLoader(
        training_set, batch_size=batch_size, shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        testing_set, batch_size=batch_size, shuffle=False,
    )
    # val_loader = torch.utils.data.DataLoader(testing_set,sampler=ImbalancedDatasetSampler(testing_set, callback_get_label=lambda x, i: tuple(x[i][1].tolist())),batch_size=batch_size,)

    logging.info("Dataset and Dataloaders created")
    # create a model
    extractor_name = cfg["train"]["extractor"]
    model = cls.ClassificationModel(model_name=extractor_name).create_model()
    #load checkpoint to continue training
    if checkpoint is not None:
        print('...Load checkpoint from {}'.format(checkpoint))
        checkpoint = torch.load(checkpoint)
        model.load_state_dict(checkpoint)
        print('...Checkpoint loaded')
        
        classifier = nn.Sequential(
            nn.Linear(1408, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, 6, bias=True)
        )
        
        # create classfier
        # replace the last linear layer with your custom classifier
        # model._avg_pooling = SPPLayer([1,2,4])
        model._fc = classifier
        # model.last_linear = self.cls
        # select with layers to unfreeze
        params  = list(model.parameters())
        len_param = len(params)
        # for index,param in enumerate(model.parameters()):
        #     if index == (len_param -1):
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False
        # for param in model.parameters():
        #     print(param.requires_grad)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: {} ".format(device))
    # convert to suitable device
    # global model
    model = model.to(device)
    logging.info("Model created...")

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
            custom_loss,
            loss_function,
            "The loss {} is not available".format(loss_function),
        )
    criterion = criterion()  
    optimizer = getattr(
        torch.optim, optimizers, "The optimizer {} is not available".format(optimizers)
    )
    max_lr = 3e-3  # Maximum LR
    min_lr = 1e-5  # Minimum LR
    t_max = 10     # How many epochs to go from max_lr to min_lr
    # optimizer = torch.optim.Adam(
    # params=model.parameters(), lr=max_lr, amsgrad=False)
    optimizer = optimizer(model.parameters(), lr=learning_rate)
    save_method = cfg["train"]["lr_scheduler_factor"]
    patiences = cfg["train"]["patience"]
    lr_factor = cfg["train"]["reduce_lr_factor"]
    # scheduler = ReduceLROnPlateau(
    #     optimizer, save_method, patience=patiences, factor=lr_factor
    # )
    scheduler = CosineAnnealingLR(
    optimizer, T_max=t_max, eta_min=min_lr)

    # before training, let's create a file for logging model result
    
    log_file = logger.make_file(cfg["session"]["sess_name"], time_str)
    

    logger.log_initilize(log_file)
    print("Beginning training...")
    # export the result to log file
    f = open("saved/logs/traning_{}.txt".format(cfg["session"]["sess_name"]), "a")
    logging.info("-----")
    logging.info("session name: {} \n".format(cfg["session"]["sess_name"]))
    logging.info(model)
    logging.info("\n")
    logging.info("CONFIGS \n")
    # logging the configs:
    # logging.info(f.read())
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
        
        logging.info(
            "Epoch {} / {} \n Training loss: {} - Other training metrics: ".format(
                i + 1, num_epoch, loss
            )
        )
        print( "Epoch {} / {} \n Training acc: {} - Other training metrics: ".format(
                i + 1, num_epoch, train_result["accuracy_score"]
            ))
        print("Epoch {} / {} \n Training loss: {} - Other training metrics: ".format(
                i + 1, num_epoch, loss
            ))
        f.write( "Epoch {} / {} \n Training loss: {} - Other training metrics: ".format(
                i + 1, num_epoch, loss
            ))
        f.write( "Epoch {} / {} \n Training acc: {} - Other training metrics: ".format(
                i + 1, num_epoch, train_result["accuracy_score"]
            ))
        tensorboard_writer.add_scalar("training accuracy",train_result["accuracy_score"],i + 1)
        tensorboard_writer.add_scalar("training f1_score",train_result["f1_score"],i + 1)
        tensorboard_writer.add_scalar("training metrics",loss,i + 1)
        logging.info(train_result)
        logging.info(
            " \n Validation loss : {} - Other validation metrics:".format(val_loss)
        )
        print("Epoch {} / {} \n valid acc: {} - Other training metrics: ".format(
                i + 1, num_epoch, val_result["accuracy_score"]
            ))
        f.write(  " \n Validation loss : {} - Other validation metrics:".format(val_loss))
        tensorboard_writer.add_scalar("valid accuracy",val_result["accuracy_score"],i + 1)
        tensorboard_writer.add_scalar("valid f1_score",val_result["f1_score"],i + 1)
        tensorboard_writer.add_scalar("valid metrics",val_loss,i + 1)
        logging.info(val_result)
        logging.info("\n")
        # saving epoch with best validation accuracy
        if best_val_acc < float(val_result["accuracy_score"]):
            logging.info(
                "Validation accuracy= "+
                str(val_result["accuracy_score"])+
                "===> Save best epoch"
            )
            f.write(  "Validation accuracy= "+
                str(val_result["accuracy_score"])+
                "===> Save best epoch")
            best_val_acc = val_result["accuracy_score"]
            torch.save(
                model.state_dict(),
                "saved/models/" + time_str + "-" + cfg["train"]["save_as_name"],
            )
        scheduler.step(val_loss)
        # else:
        #     # logging.info(
        #     #     "Validation accuracy= "+ str(val_result["accuracy_score"])+ "===> No saving"
        #     # )
        #     continue


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
    test_loader = torch.utils.data.DataLoader(testing_set, batch_size=32, shuffle=False)
    print("Inference on the testing set")

    # load the test model and making inference
    test_model = cls.ClassificationModel(model_name=extractor_name).create_model()
    model_path = os.path.join(
        "saved/models", time_str + "-" + cfg["train"]["save_as_name"]
    )
    test_model.load_state_dict(torch.load(model_path))
    test_model = test_model.to(device)
    logging.info(tester.test_result(test_model, test_loader, device,cfg))

    # saving torch models


if __name__ == "__main__":
    main()
