import pandas as pd
import shutil
import os
def main():
    
    training_set = pd.read_csv("dataset/train3k_fold_0.csv", usecols=["file_name", "label"])

    for i in training_set["file_name"].unique().tolist():
        shutil.copyfile("/root/data/{}".format(i), "/root/pytorch/torchbox/subdata/{}_{}".format(i.split('/')[-2],i.split('/')[-1])) 


if __name__ == "__main__":
    main()