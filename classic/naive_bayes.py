import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import random_split, DataLoader
import random, numpy as np
import torch.nn.functional as F

## SETUP
SEED = 42
BATCH_SIZE = 128
THRESHOLD = 0.5
NUM_CLASSES = 10
DEVICE = "cuda:0"
ALPHA = 1.0
FEATURE_DIM = 784
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed(SEED)

## Data
def mnist_loaders(batch_size=128,root="./data"):
    def binarize(x,threshold): return (x>threshold).float()
    imt = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambd=lambda x: binarize(x,THRESHOLD)),
        transforms.Lambda(lambd=lambda x : x.view(-1)),
    ])
    full_train = datasets.MNIST(root,train=True,transform=imt,download=True)
    train,val = random_split(full_train,[0.8,0.2],generator=torch.Generator().manual_seed(SEED))
    test = datasets.MNIST(root,train=False,transform=imt,download=True)
    train_loader = DataLoader(dataset=train,batch_size=batch_size,shuffle=True,pin_memory=True,num_workers=4)
    val_loader = DataLoader(dataset=val,batch_size=batch_size,shuffle=False,pin_memory=True,num_workers=4)
    test_loader = DataLoader(dataset=test,shuffle=False,batch_size=batch_size,pin_memory=True,num_workers=4)
    return train_loader,val_loader,test_loader
train_loader,val_loader,test_loader = mnist_loaders(BATCH_SIZE)

## Training
# Step 1: log P(y)
with torch.no_grad():
    label_counts = torch.zeros(NUM_CLASSES,dtype=torch.float32).to(device=DEVICE)
    for _,y in train_loader:
        y = y.to(device=DEVICE)
        label_counts += torch.bincount(input=y,minlength=NUM_CLASSES).to(torch.float32)
total_smoothed = label_counts.sum()+(ALPHA*NUM_CLASSES)
label_probs = torch.log((label_counts+ALPHA)/total_smoothed)
# Step 2: SIGMA_i log P(x_i|y)
with torch.no_grad():
    feat_pos = torch.zeros(NUM_CLASSES, FEATURE_DIM, dtype=torch.float32, device=DEVICE)
    for x,y in train_loader:
        x,y = x.to(device=DEVICE),y.to(device=DEVICE)
        y1h = F.one_hot(y, NUM_CLASSES).to(torch.float32)
        feat_pos += y1h.T @ x
    denom = label_counts.view(-1, 1) + 2.0 * ALPHA
    cond_p = (feat_pos + ALPHA) / denom
    log_p0 = torch.log((1.0 - cond_p).clamp_min(1e-8))
    log_p1 = torch.log(cond_p.clamp_min(1e-8))
    
## Inference
@torch.no_grad()
def predict_batch(xb):
    ll = xb @ log_p1.T + (1.0 - xb) @ log_p0.T
    ll = ll + label_probs.unsqueeze(0)
    return ll.argmax(dim=1)
@torch.no_grad()
def evaluate(loader):
    correct, total = 0, 0
    for xb, yb in loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)
        pred = predict_batch(xb)
        correct += (pred == yb).sum().item()
        total += yb.size(0)
    return correct / total

val_acc  = evaluate(val_loader)
test_acc = evaluate(test_loader)
print(f"NB Val Acc: {val_acc:.4f}  Test Acc: {test_acc:.4f}")

@torch.no_grad()
def confusion_matrix(loader):
    cm = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64)
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        pred = predict_batch(xb)
        for t, p in zip(yb.cpu().tolist(), pred.cpu().tolist()):
            cm[t, p] += 1
    return cm

cm = confusion_matrix(test_loader)
print(cm)