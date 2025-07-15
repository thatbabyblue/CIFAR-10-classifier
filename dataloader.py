from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(batch_size=64):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5))])

    train_set = datasets.CIFAR10(root='/Users/seanwayne/PycharmProjects/learn_pytorch/dataset1/cifar-10-batches-py', train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(root='/Users/seanwayne/PycharmProjects/learn_pytorch/dataset1/cifar-10-batches-py',
                                 train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader