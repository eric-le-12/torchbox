from torchvision import transforms
import torch
# from torchtoolbox.transform.cutout import Cutout_PIL as Cutout
# define augmentation methods for training and validation/test set
train_transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.CenterCrop(224),
        transforms.RandomPerspective(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ]
)
# train_transform = transform=transforms.Compose([
#                                         transforms.RandomCrop(224),
#                                         transforms.RandomHorizontalFlip(),
#                                         transforms.ToTensor(),
#                                         transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
# ])
# val_transform = transforms.Compose(
#     [transforms.Resize((256, 256)),transforms.CenterCrop(224), transforms.ToTensor(),]
# )
val_transform = transform=transforms.Compose([
							transforms.Resize((256, 256)),
							transforms.CenterCrop(224),
							transforms.ToTensor(),
							transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
							])