import torch.nn as nn
import torch
import os
from sklearn.metrics import classification_report
from data_loader import dataloader
import pandas as pd
import json
from data_loader import transform
from model import classification as cls

# use this file if you want to quickly test your model




def test_result(model, test_loader, device,cfg):
    # testing the model by turning model "Eval" mode
    model.eval()
    preds = []
    labels = []
    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device).float()
        with torch.no_grad():
            output = model(data)
            output_sigmoid = torch.sigmoid(output)
            preds = (output_sigmoid.cpu().numpy() >0.5).astype(float)
        labels = target.cpu().numpy()      
    return (classification_report(labels, preds, target_names=cfg["data"]["label_dict"]))


def main():
    print("Testing process beginning here....")


if __name__ == "__main__":
    main()
    with open("cfgs/chexphoto.cfg") as f:
        cfg = json.load(f)
    test_data = cfg["data"]["test_csv_name"]
    data_path = cfg["data"]["data_path"]
    test_df = pd.read_csv(test_data, usecols=["file_name", "label"])
    # prepare the dataset
    testing_set = dataloader.ClassificationDataset(
        test_df, data_path, transform.val_transform
    )
    # make dataloader
    test_loader = torch.utils.data.DataLoader(testing_set, batch_size=1, shuffle=False,)
    # load model
    extractor_name = cfg["train"]["extractor"]
    model = cls.ClassificationModel(model_name=extractor_name).create_model()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join("saved/models", cfg["train"]["save_as_name"])
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)

    # print classification report
    test_result(model, test_loader, device,cfg)
