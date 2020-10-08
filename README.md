# torchbox - a fully customizable deep learning implementation solution in Pytorch
implement and fine-tune your deep learning model in an easy, customizable and fastest way 

@Lexuanhieu131297 - First version created by Oct 2019

# Introduction

Several problems that deep learning researchers usually face are that: 

1. Everytime they train a model, they have to replicate or copy&paste many 
resuable code, which is time-expensive

2. The code is unstructed and unclean, making it more difficult to read and interpret

3. The work is messy and hardly replicable. 

4. To few metrics

I therefore release torchbox, a Deep Learning implementation tool for students, and scientists that can tackle these aforementioned problems.
torchbox provides you a fast, clean, structured and repeatable implementation and fine-tuning of your custom Deep Learning models. Moreoever, you can add as many metrics as you can to measure your model, all you need to provide is the metric's class and its methods. For example, we already integrated all of Sklearn metrics, all you need to do is provide the metric method names, example: f1_score, precision_score, etc... No more painful metrics coding!!!

# What it provides

### Modularity and customizability

Every components you need in training is a separated module. 

### Scalable project

### All in one config file

Introduce the usage of a config file that contains neccessary params for training. Everything you need is just modify this file in order to train

### Simple training and testing 

Only ``` python train.py```  is required to do so

### Specilized for transfer learning

I have written a class wrapper that can make it easier for you to turn on transfer learning. Instead of importing a pretrained model, excluding
its head, all you need is just writting down it names and customize your new classifer (head).

List of models are available here :

### Attach your custom metrics easily

Suppose you have implemented a custom metrics, for example the Thresholded Jaccard Similarity score in the ISIC 2019 Segmentation competition, you can use it to measure your model without the pain of editting/typing too much code. Simply inputing the metric class name and its method to use

# Quick start

## Example Usage

I have already implemented a built-in classification example. To use it:

1. Starting by cloning this github into your machine

2. CD to the project folder and install dependencies 

3. Editting config file as needed (See documentation)

4. Run : `python train.py`

You should expect to see sthing like this:

![](https://i.imgur.com/UyQ6mzK.png)

The log files for training is available in saved/logs folder

# Documentation

## The config files
location : cfg/tenes.cfg

The config file is the core file in this package. It contains core parameters to train. While most of them are interpretable, I provide here docs for some important params.

### sess_name
```json
{
  "session": {
    "sess_name" : "trial_1"
  },
```
The sess_name should be unique and is an identifier of your training model

### data params
```json
    "data_csv_name": "dataset/train.csv",
    "validation_ratio": "0.2",
    "test_csv_name": "dataset/test.csv",
    "data_path": "dataset/ex_data/",
    "label_dict": ["cat","dog"]
```
In order to train, we requires two csv files for the training and testing set, each has **two columns named "file_name" and "label (int starting from 0)"**

Kindly provide the two csv file path in "data_csv_name" (for training set) and "test_csv_name" (for testing set)

All of your images should be inside the folder indicated by the param **data_path**

The training set will be further splitted to a smaller validation set by the **validation_ratio**

label_dict: a list of all label names for mapping

### Optimizer
    "name": "Adam",
    "lr": 0.0002,
    "loss": "CrossEntropyLoss"

"name" : name of the optimizer class in torch.optim. Ex: if you intend to use torch.optim.Adam simply type "Adam"

Similarly, if you intend to use torch.nn. CrossEntropyLoss simply type CrossEntropyLoss

### Model
    "extractor": "resnet18",
	  "metrics": ["accuracy_score","f1_score"],

The pretrained architecture to be used as feature extractor and the metrics 

## Customizing your metrics: use your own metrics

The **utils/metrics.py** file is a wrap file that handles the metrics implementation

Let's say use want to use your custom and newly created metrics. All you need to do is implement a Class which has a method to take in (labels [numpy or list type],preds) and calculate your measurement.

Example: MyMetrics.IoU(labels,preds) 

Then in the utils/metrics.py file: 
1. import your class

```python

            do_metric = getattr(
                skmetrics, metric, "The metric {} is not implemented".format(metric)
            )
```


2.Replace the skmetrics with your class, for ex: MyMetrics

3.Editing metrics name in your config file, example: `"metrics" : ["IoU"]`

## Customizing your model

In the **model/classification.py** file:

Feel free to edit the `class ClassificationModel` but make sure your desired model's architecture is returned by the **create_model** method

## Customizing tranformation(augmentation) and dataloaders

See data_loader/dataloader.py for customizing dataloaders
See data_loader/transform.py for customizing the transformation

## Customizing training actions:

See trainer.py, the file containing scripts for training one entire epoch.

## Making new projects 

To make a new project with neccessary files:

`python new_project ../new-project
`





