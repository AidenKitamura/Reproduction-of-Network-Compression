import torch
from torchvision.models import resnet18
import data_loader_l1norm

def eval(model, test_loader):
    '''Evaluates the model and produces accuracy results.

    Args:
        model: a instance of class ResNet18
        test_loader: A dataloader that provides the whole test dataset
            in terms of torch tensor and grouped batch by batch

    Returns:
        A float that provides the accuracy of the model on CIFAR-10
    '''
    correct = 0
    total = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    with torch.no_grad():
        for i, (img, target) in enumerate(test_loader):
            img = img.to(device)
            out = model(img)
            pred = out.max(1)[1].detach().cpu().numpy()
            target = target.cpu().numpy()
            correct += (pred==target).sum()
            total += len(target)
    return correct / total

if __name__ == '__main__':
    model = resnet18(pretrained=True)
    train_loader, test_loader = data_loader_l1norm.get_dataloader()
    
    correct = 0
    total = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    with torch.no_grad():
        for i, (img, target) in enumerate(test_loader):
            img = img.to(device)
            out = model(img)
            pred = out.max(1)[1].detach().cpu().numpy()
            target = target.cpu().numpy()
            correct += (pred==target).sum()
            total += len(target)
            break
    acc = correct / total
    print("Acc=%.4f\n"%(acc))