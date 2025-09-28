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
OUTPUT_DIM = 10
DEVICE = "cuda:0"
BATCH_SIZE = 16
C = 5.0
LR = 1e-2
REG = 0.5
NUM_EPOCHS = 10
SAVE_PATH = 'mini_implementations/models/multi_svm.pth'
set_seed(SEED)

## Data
tfm = transforms.ToTensor()
train_full = MNIST("data/MNIST", train=True, transform=tfm, download=True)
g = torch.Generator().manual_seed(SEED)
train_ds, val_ds = random_split(train_full, [int(0.8*len(train_full)), len(train_full) - int(0.8*len(train_full))], generator=g)
test = MNIST("data/MNIST", train=False, transform=tfm, download=True)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
test_loader  = DataLoader(test,     batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

## Model
svm = nn.Linear(INPUT_DIM,OUTPUT_DIM,bias=True)
svm.to(device=DEVICE)
opt = optim.SGD(svm.parameters(), lr=LR)

@torch.no_grad()
def eval_loader(loader):
    svm.eval()
    correct, total = 0, 0
    for x, y in loader:
        x = x.to(DEVICE).view(x.size(0), -1) 
        y = y.to(DEVICE).long()
        scores = svm(x)
        pred = scores.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / total

## Training
for epoch in tqdm(range(NUM_EPOCHS),desc="Training : "):
    svm.train()
    running = 0.0
    for x,y in train_loader:
        x = x.to(DEVICE).view(x.size(0), -1)
        y = y.to(DEVICE).long().view(-1, 1)
        opt.zero_grad(set_to_none=True)
        scores = svm(x)
        reg = REG * torch.sum(svm.weight ** 2)
        correct = torch.gather(scores,1,y.view(-1,1))
        margins = scores - correct + 1
        margins.scatter_(1,y.view(-1,1),0.0)
        hinge_loss = torch.square(torch.clamp(margins,min=0)).sum(dim=1).mean()
        loss = reg +  C* hinge_loss
        loss.backward()
        opt.step()
        running += loss.item() * x.size(0)
    val_acc = eval_loader(val_loader)
    print(f"Epoch {epoch+1}: train_loss={(running/len(train_loader.dataset)):.4f}  val_acc={val_acc:.4f}")
torch.save(svm.state_dict(),SAVE_PATH)

test_acc = eval_loader(test_loader)
print(f"test_acc={test_acc}")