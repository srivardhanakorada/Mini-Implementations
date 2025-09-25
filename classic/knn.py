## Need to code tiled version in future

import torch
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import random_split
from torch.utils.data import DataLoader, Subset
import time

## Setup
TRAIN_SIZE = 12500
SEED = 42
BATCH_SIZE = 100
k_start = 3
DEVICE = "cuda:0"
PCA_MIN = 10
PCA_MAX = 50
PCA_INT = 10

## Data
tfm = transforms.ToTensor()
train_full = MNIST("mini_implementations/data/MNIST",train=True,transform=tfm,download=True)
test = MNIST("mini_implementations/data/MNIST",train=False,transform=tfm,download=True)
idx = torch.randperm(len(train_full),generator=torch.Generator().manual_seed(SEED))
selected_idx = idx[:TRAIN_SIZE]
train_small = Subset(train_full,selected_idx.tolist())
train,val = random_split(train_small,[0.8,0.2],generator=torch.Generator().manual_seed(SEED))
train_loader = DataLoader(train,batch_size=BATCH_SIZE,shuffle=True,num_workers=4,pin_memory=True)
val_loader = DataLoader(val,batch_size=BATCH_SIZE,shuffle=False,num_workers=4,pin_memory=True)
test_loader = DataLoader(test,batch_size=BATCH_SIZE,shuffle=False,num_workers=4,pin_memory=True)

## Model
## Same as a loader

## Training
x_train,y_train = [],[]
for x,y in train_loader:
    x_train.append(x.view(x.size(0),-1))
    y_train.append(y)
x_train = torch.cat(x_train,dim=0).to(device=DEVICE)
y_train = torch.cat(y_train,dim=0).to(device=DEVICE)

## Testing
print("******* k-varying ********")
for k in range(k_start,7):
    test_crct,total = 0,0
    start = time.time()
    for x,y in test_loader:
        x,y = x.view(x.size(0),-1).to(device=DEVICE),y.to(device=DEVICE)
        dist = torch.cdist(x,x_train)
        vals,nn_idx = dist.topk(k=k,largest=False,dim=1)
        nn_labels = y_train[nn_idx]
        preds = torch.mode(nn_labels,dim=1).values
        test_crct += (preds == y).sum().item()
        total += y.size(0)
    end = time.time()
    print(f"k = {k} : Acc = {test_crct/total}, Time = {end-start}")

def pca_fit(X,n_comp):
    mean = X.mean(dim=0, keepdim=True) 
    Xc = X - mean
    _,_,V = torch.pca_lowrank(Xc,q=n_comp,center=False)
    return mean,V[:,:n_comp]

def pca_transform(X,mean,dirs):
    return (X-mean) @ dirs

mean, V = pca_fit(x_train, PCA_MAX)
proj_trains = {}
for d in range(PCA_MIN,PCA_MAX+1,PCA_INT): proj_trains[d] = pca_transform(x_train,mean,V[:,:d])
print("******* d-varying ********")
for d in range(PCA_MIN,PCA_MAX+1,PCA_INT):
    test_crct,total = 0,0
    x_train_min = proj_trains[d]
    start = time.time()
    for x,y in test_loader:
        x,y = x.view(x.size(0),-1).to(device=DEVICE),y.to(device=DEVICE)
        x_pca = pca_transform(x,mean,V[:,:d])
        dist = torch.cdist(x_pca,x_train_min)
        vals,nn_idx = dist.topk(k=5,largest=False,dim=1)
        nn_labels = y_train[nn_idx]
        preds = torch.mode(nn_labels,dim=1).values
        test_crct += (preds == y).sum().item()
        total += y.size(0)
    end = time.time()
    print(f"d = {d} : Acc = {test_crct/total}, Time = {end-start}")