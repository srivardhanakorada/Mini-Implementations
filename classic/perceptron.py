import torch
from mini_implementations.common.utils import set_seed #type:ignore
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import random_split
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

## Setup
SEED = 42
INPUT_DIM = 784
OUTPUT_DIM = 1
DEVICE = "cuda:0"
BATCH_SIZE = 1
LR = 1
NUM_EPOCHS = 10
SAVE_PATH = 'mini_implementations/models/perceptron.pth'
set_seed(SEED)

## Data
tfm = transforms.ToTensor()
custom_tfm = transforms.Compose([
    transforms.Lambda(lambd=lambda y: -1 if y == 0 else y)
])
train_full = MNIST("data/MNIST",train=True,transform=tfm,download=True,target_transform=custom_tfm)
idx = (train_full.targets == -1) | (train_full.targets == 1)
train_full.data,train_full.targets = train_full.data[idx],train_full.targets[idx]
g = torch.Generator().manual_seed(SEED)
train,val = random_split(train_full,[0.8,0.2],generator=g)
train_loader = DataLoader(train,batch_size=BATCH_SIZE,shuffle=True,num_workers=4,pin_memory=True)
val_loader = DataLoader(val,batch_size=BATCH_SIZE,shuffle=False,num_workers=4,pin_memory=True)
test = MNIST("data/MNIST",train=False,transform=tfm,download=True,target_transform=custom_tfm)
idx = (test.targets == -1) | (test.targets == 1)
test.data,test.targets = test.data[idx],test.targets[idx]
test_loader = DataLoader(test,batch_size=BATCH_SIZE,shuffle=False,num_workers=4,pin_memory=True)

## Model
perceptron = nn.Linear(INPUT_DIM,OUTPUT_DIM,bias=False)
perceptron.to(device=DEVICE)

## Training
for epoch in tqdm(range(NUM_EPOCHS),desc="Training : "):
    flag = True
    for x,y in train_loader:
        x = x.to(device=DEVICE)
        y = y.to(device=DEVICE)
        x = x.flatten().view(1,-1)
        y = y.view(1,1)
        with torch.no_grad():
            if perceptron(x)*y < 0: 
                flag = False
                perceptron.weight.add_(y*x)
    if flag:break
torch.save(perceptron.state_dict(),SAVE_PATH)

## Testing
test_crct = 0
for x,y in test_loader:
    x = x.to(device=DEVICE)
    y = y.to(device=DEVICE)
    x = x.flatten()
    x = x.flatten().view(1,-1)
    y = y.view(1,1)
    y_pred = perceptron(x)
    test_crct += (y == torch.sign(y_pred)).sum().item()
print(f"Test Accuracy : {test_crct/len(test_loader)}")