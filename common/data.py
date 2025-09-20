import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def mnist_loaders(batch_size=128,root="./data"):
    imt = transforms.ToTensor()
    train = datasets.MNIST(root,train=True,transform=imt,download=True)
    test = datasets.MNIST(root,train=False,transform=imt,download=True)
    train_loader = DataLoader(dataset=train,batch_size=batch_size,shuffle=True,pin_memory=True,num_workers=4)
    test_loader = DataLoader(dataset=test,shuffle=False,batch_size=batch_size,pin_memory=True,num_workers=4)
    return train_loader,test_loader