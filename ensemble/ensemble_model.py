import torch
import torch.nn as nn
class MyEnsemble(nn.Module):
    def __init__(self, modelA, modelB,modelC, nb_classes=10):
        super(MyEnsemble, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        # Remove last linear layer
        self.modelA.fc = nn.Identity()
        self.modelB.fc = nn.Identity()
        
        # Create new classifier
        self.classifier = nn.Linear(2048+512, nb_classes)
        
    def forward(self, x):
        x1 = self.modelA(x.clone())  # clone to make sure x is not changed by inplace methods
        x1 = x1.view(x1.size(0), -1)
        x2 = self.modelB(x)
        x2 = x2.view(x2.size(0), -1)
        x = torch.cat((x1, x2), dim=1)       
        x = self.classifier(F.relu(x))
        return x