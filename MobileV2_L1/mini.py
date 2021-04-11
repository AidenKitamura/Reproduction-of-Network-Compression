import torch
from torchvision.models import resnet18
import torch_pruning as tp
import sys
sys.path.insert(0, 'examples/')
import data_loader_l1norm


def eval(model, test_loader):
    correct = 0
    total = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    with torch.no_grad():
        for i, (img, target) in enumerate(test_loader):
            img = img.to(device)
            out = model(img)
            print(type(out))
            pred = out.max(1)[1].detach().cpu().numpy()
            target = target.cpu().numpy()
            correct += (pred==target).sum()
            total += len(target)
    return correct / total

if __name__ == '__main__':
    # # 1. setup strategy (L1 Norm)
    # strategy = tp.strategy.L1Strategy() # or tp.strategy.RandomStrategy()

    # # 2. build layer dependency for resnet18
    # DG = tp.DependencyGraph()
    # DG.build_dependency(model, example_inputs=torch.randn(1,3,224,224))

    # # 3. get a pruning plan from the dependency graph.
    # pruning_idxs = strategy(model.conv1.weight, amount=0.4) # or manually selected pruning_idxs=[2, 6, 9]
    # pruning_plan = DG.get_pruning_plan( model.conv1, tp.prune_conv, idxs=pruning_idxs )
    # print(pruning_plan)

    # # 4. execute this plan (prune the model)
    # pruning_plan.exec()

    # 5. eval
    # acc = eval(model, test_loader)
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