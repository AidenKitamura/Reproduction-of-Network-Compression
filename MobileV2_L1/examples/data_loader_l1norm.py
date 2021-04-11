import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10


def get_dataloader():
    '''Load the dataset from CIFAR10.

    Returns:
        train_loader:A dataloader that provides the whole train dataset
            in terms of torch tensor and grouped batch by batch
        test_loader: A dataloader that provides the whole test dataset
            in terms of torch tensor and grouped batch by batch
    '''
    train_loader = torch.utils.data.DataLoader(
        CIFAR10('./data', train=True, transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]), download=True),batch_size=1, num_workers=2)
    test_loader = torch.utils.data.DataLoader(
        CIFAR10('./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
        ]),download=True),batch_size=1, num_workers=2)
    return train_loader, test_loader

if __name__=='__main__':
    train_loader, test_loader = get_dataloader()
    print(type(train_loader),len(train_loader))
    for i, data in enumerate(train_loader, 0):
        print(i)
        # PIL
        img = transforms.ToPILImage()(data[i][0])
        img.show()
        break