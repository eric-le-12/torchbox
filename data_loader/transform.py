from torchvision import transforms
from torchtoolbox.transform.cutout import Cutout_PIL as Cutout
# define augmentation methods for training and validation/test set

train_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomPerspective(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
    ]
)

val_transform = transforms.Compose(
    [transforms.Resize((128, 128)), transforms.ToTensor(),]
)
