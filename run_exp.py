"""
Note that one can use either this file or its deprecated version train.py
New version supports : 
* Tracking experiment data and checkpoint by using neptune.ai
* Support K Fold cross validation
* Support different class of loading data: time series vs images data
* Support custom collocation and running script for time series data
* Support running from 3 different subsets instead of 2 in previous version

"""
import re
import importlib
import json
import time
import utils
from model import classification as model
import data_loader
# from data_loader import dataloader
from data_loader.dataloader import data_split
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
# from torchsampler import ImbalancedDatasetSampler


def main(collocation,model,dataset,validation_flag,current_fold,comment="No comment", checkpoint=None,logger=None):
    # parser = argparse.ArgumentParser(description='NA')
    # parser.add_argument('-c', '--configure', default='cfgs/chexphoto.cfg', help='JSON file')
    # parser.add_argument('-cp', '--checkpoint', default=None, help = 'checkpoint path')
    # args = parser.parse_args()
    # checkpoint = args.checkpoint
    # # read configure file
    # with open(args.configure) as f:
    #     cfg = json.load(f)
    # using parsed configurations to create a dataset
    # read training set
    
    data = cfg["data"]["data_csv_name"]
    data = re.sub(r"fold[0-9]",str(current_fold),data)
    print("Reading training data from file: ",data)
    training_set = pd.read_csv(data,delimiter='*',header=None)
    
    # check if validation flag is on
    if (validation_flag==1):
        # using custom validation set
        print("Creating validation set from file")
        valid = cfg["data"]["validation_csv_name"]
        valid = re.sub(r"fold[0-9]",str(current_fold),valid)
        print("Reading validation data from file: ",valid)
        valid_set = pd.read_csv(valid,delimiter='*',header=None)
    else:
        # auto divide validation set
        validation_split = float(cfg["data"]["validation_ratio"])
        training_set,valid_set = data_split(training_set,validation_split)

    data_path = cfg["data"]["data_path"]
    batch_size = int(cfg["data"]["batch_size"])

    
    # create dataset
    # train, test, _, _ = dataloader.data_split(training_set, validation_split)

    training_set = dataset(
        training_set, data_path, padding=True,normalize=True
    )

    testing_set = dataset(
        valid_set, data_path, padding=True,normalize=True
    )
    # create dataloaders
    # global train_loader
    # global val_loader
    # SAmpler to prevent inbalance data label
    # train_loader = torch.utils.data.DataLoader(training_set,sampler=ImbalancedDatasetSampler(training_set, callback_get_label=lambda x, i: tuple(x[i][1].tolist())),batch_size=batch_size,)

    # End sampler
    train_loader = torch.utils.data.DataLoader(
        training_set, batch_size=batch_size, shuffle=True,collate_fn=collocation
    )
    val_loader = torch.utils.data.DataLoader(
        testing_set, batch_size=batch_size, shuffle=False,collate_fn=collocation
    )
    # val_loader = torch.utils.data.DataLoader(testing_set,sampler=ImbalancedDatasetSampler(testing_set, callback_get_label=lambda x, i: tuple(x[i][1].tolist())),batch_size=batch_size,)

    logging.info("Dataset and Dataloaders created")
    # create a model
    # extractor_name = cfg["train"]["extractor"]
    # model = cls(model_name=extractor_name).create_model()
    model = cls(class_num=2,num_of_blocks=9,training=True,dense_layers=[256,256])
    # load checkpoint to continue training
    if checkpoint is not None:
        print("...Load checkpoint from {}".format(checkpoint))
        checkpoint = torch.load(checkpoint)
        model.load_state_dict(checkpoint)
        print("...Checkpoint loaded")

        classifier = nn.Sequential(
            nn.Linear(1408, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, 6, bias=True),
        )

        # create classfier
        # replace the last linear layer with your custom classifier
        # model._avg_pooling = SPPLayer([1,2,4])
        # model._fc = classifier
        # model.last_linear = self.cls
        # select with layers to unfreeze
        params = list(model.parameters())
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
    print(sum(p.numel() for p in model.parameters()))
    time.sleep(4)

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
    criterion = custom_loss.WeightedFocalLoss(weight=None, gamma=2,reduction='sum')
    optimizer = getattr(
        torch.optim, optimizers, "The optimizer {} is not available".format(optimizers)
    )
    max_lr = 3e-3  # Maximum LR
    min_lr = 1e-5  # Minimum LR
    t_max = 10  # How many epochs to go from max_lr to min_lr
    # optimizer = torch.optim.Adam(
    # params=model.parameters(), lr=max_lr, amsgrad=False)
    optimizer = optimizer(model.parameters(), lr=learning_rate)
    save_method = cfg["train"]["lr_scheduler_factor"]
    patiences = cfg["train"]["patience"]
    lr_factor = cfg["train"]["reduce_lr_factor"]
    # scheduler = ReduceLROnPlateau(
    #     optimizer, save_method, patience=patiences, factor=lr_factor
    # )
    scheduler = ReduceLROnPlateau(optimizer, mode='min',factor=0.5,min_lr=0.00001,verbose=True,patience=3)

    # before training, let's create a file for logging model result


    print("Beginning training...")
    time.sleep(3)
    # export the result to log file
    logging.info("-----")
    logging.info("session name: {} \n".format(cfg["session"]["sess_name"]))
    logging.info("session description: {} \n".format(comment))
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
        # tensorboard_writer.add_scalar("training accuracy",train_result["accuracy_score"],i + 1)
        # tensorboard_writer.add_scalar("training f1_score",train_result["f1_score"],i + 1)
        # tensorboard_writer.add_scalar("training metrics",loss,i + 1)
        logging.info(train_result)
        logging.info(
            " \n Validation loss : {} - Other validation metrics:".format(val_loss)
        )
        
        # tensorboard_writer.add_scalar("valid accuracy",val_result["accuracy_score"],i + 1)
        # tensorboard_writer.add_scalar("valid f1_score",val_result["f1_score"],i + 1)
        # tensorboard_writer.add_scalar("valid metrics",val_loss,i + 1)
        logging.info(val_result)
        logging.info("\n")
        # saving epoch with best validation accuracy
        if best_val_acc < float(val_result["f1_score"]):
            logging.info(
                "Validation f1= "
                + str(val_result["f1_score"])
                + "===> Save best epoch \n"
            )

            best_val_acc = val_result["f1_score"]
            torch.save(
                model.state_dict(),
                "saved/models/" + time_str + "-" + str(current_fold) + "-" +cfg["train"]["save_as_name"],
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
    test_data = re.sub(r"fold[0-9]",str(current_fold),test_data)
    print("reading testing data from file: ",test_data)
    test_df = pd.read_csv(test_data,delimiter='*',header=None)
    

    # prepare the dataset
    testing_set = dataset(
        test_df, data_path, padding=False, normalize=True
    )

    # make dataloader
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_loader = torch.utils.data.DataLoader(testing_set, batch_size=1, shuffle=False,collate_fn=collocation)
    print("Inference on the testing set")

    # load the test model and making inference
    test_model = cls(class_num=2,num_of_blocks=9,training=True,dense_layers=[256,256])
    model_path = os.path.join(
        "saved/models", time_str + "-" + str(current_fold) + "-" + cfg["train"]["save_as_name"]
    )
    test_model.load_state_dict(torch.load(model_path))
    test_model = test_model.to(device)
    logging.info(tester.adaptive_test_result(test_model, test_loader, device, cfg))

    # saving torch models
    print("---End of testing phase----")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="NA")
    parser.add_argument(
        "-c", "--configure", default="cfgs/chexphoto.cfg", help="JSON file"
    )
    parser.add_argument("-cp", "--checkpoint", default=None, help="checkpoint path")
    args = parser.parse_args()
    checkpoint = args.checkpoint

    # read configure file
    with open(args.configure) as f:
        cfg = json.load(f)

    # comment for this experiment: leave here
    comment = "First demo comparison"
    
    # modify this part if you are using kfold
    # csv files of kfold should be in format : *_fold0.csv , _fold1.csv...
    fold_list = ["fold0", "fold1", "fold2", "fold3", "fold4"]

    # automate the validation split or not
    if float(cfg["data"]["validation_ratio"]) > 0 and cfg["data"]["validation_csv_name"] == "":
        print("No validation set available, auto split the training into validation")
        validation_flag = cfg["data"]["validation_ratio"]
    else:
        validation_flag = 1
    
    # choose way of loading data (collocation)
    module_name = cfg["data"]["collocation"]
    try:
        from utils.collocation import collocation as custom_collate
        collocation = getattr(custom_collate,module_name) 
        print("Successfully imported collocation module")
    
    except:
        print("Cannot import data collocation module".format(module_name))
    
    # choose dataloader type
    module_name = cfg["data"]["data.class"]
    
    try:
        dataset = getattr(
            data_loader.dataloader,module_name
        )
        print("Successfully imported data loader module")
    
    except:
        print("Cannot import data loader module".format(module_name))
    
    # choose model classification class
    module_name = cfg["train"]["model.class"]
    try:
        cls = getattr(model,module_name)
        print("Successfully imported model module")
    except:
        print("Cannot import model module".format(module_name))

    # create logger
    time_str = str(datetime.now().strftime("%Y%m%d-%H%M"))
    log_file = logger.make_file(cfg["session"]["sess_name"], time_str)
    logger.log_initilize(log_file)

    for fold in fold_list:
        # running at fold n
        print("Currently running on {}".format(fold))
        time.sleep(3)
        main(
            comment=comment,
            checkpoint=checkpoint,
            collocation=collocation,
            model=cls,
            dataset=dataset,
            validation_flag=validation_flag,
            current_fold=fold,
            logger= logger
        )
