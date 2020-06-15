import torch
import torch.nn as nn
import pretrainedmodels as ptm
import ssl


class ClassificationModel:
    def __init__(self, model_name, pretrained="imagenet", class_num=2):
        """Make your model by using transfer learning technique:  
        Using a pretrained model (not including the top layer(s)) as a feature extractor and 
        add on top of that model your custom classifier

        Args:
            model_name ([str]): [name of pretrained model]
            pretrained (str, optional): [using pretrained weight or not]. Defaults to "imagenet".
            class_num (int, optional): [number of target classes]. Defaults to 2.
        """
        self.model_name = model_name
        self.pretrained = pretrained
        self.class_num = class_num

    def classifier(self, in_features):
        # initilize your classifier here
        classifier = nn.Sequential(
            nn.Linear(in_features, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, self.class_num, bias=True),
        )

        # output should be a sequential instance
        self.cls = classifier

    def create_model(self):
        # load your pretrained model
        try:
            model = ptm.__dict__[self.model_name](pretrained=self.pretrained)
        except:
            ssl._create_default_https_context = ssl._create_unverified_context
            model = ptm.__dict__[self.model_name](pretrained=self.pretrained)
        # incoming features to the classifier
        in_features = model.last_linear.in_features
        # create classfier
        self.classifier(in_features)
        # replace the last linear layer with your custom classifier
        model.last_linear = self.cls
        # select with layers to unfreeze
        for param in model.parameters():
            param.requires_grad = True
        return model
