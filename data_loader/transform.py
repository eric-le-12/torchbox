from torchvision import transforms

train_transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomPerspective(),
        transforms.ToTensor(),
    ]
)

val_transform = transforms.Compose(
    [transforms.Resize((128, 128)), transforms.ToTensor(),]
)
