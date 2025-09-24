from mini_implementations.common.utils import set_seed #type:ignore
import torch,torch.nn as nn,torch.optim as optim,torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os

# Setup
SEED = 42
BATCH_SIZE = 1024
IN_FEATURES = 784
OUT_FEATURES = 10
NUM_EPOCHS = 30
LR = 1e-2
POS_INF = float("inf")
NEG_INF = float("-inf")
PRECISION = 3
DEVICE = "cuda:0"
BEST_MODEL_CHECKPOINT = 'mini_implementations/models/best_log_res.pth'
LATEST_MODEL_CHECKPOINT = 'mini_implementations/models/latest_log_res.pth'
LOSS_CURVE_SAVE_PATH = 'mini_implementations/results/plots/log_res_train_loss.png'
set_seed(SEED)

# Data
def mnist_loaders(batch_size=128,root="./data"):
    imt = transforms.ToTensor()
    g = torch.Generator().manual_seed(SEED)
    full_train = datasets.MNIST(root,train=True,transform=imt,download=True)
    train,val = random_split(full_train,[0.8,0.2],generator=g)
    test = datasets.MNIST(root,train=False,transform=imt,download=True)
    train_loader = DataLoader(dataset=train,batch_size=batch_size,shuffle=True,pin_memory=True,num_workers=4)
    val_loader = DataLoader(dataset=val,batch_size=batch_size,shuffle=False,pin_memory=True,num_workers=4)
    test_loader = DataLoader(dataset=test,shuffle=False,batch_size=batch_size,pin_memory=True,num_workers=4)
    return train_loader,val_loader,test_loader
train_loader,val_loader,test_loader = mnist_loaders(BATCH_SIZE)

# Model
log_reg = nn.Sequential(nn.Flatten(),nn.Linear(IN_FEATURES,OUT_FEATURES))
log_reg = log_reg.to(DEVICE)

# Training
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(log_reg.parameters(),lr=LR)
train_loss,val_loss = [],[]
best_val_acc = NEG_INF
epoch_start = 0
if os.path.exists(LATEST_MODEL_CHECKPOINT):
    checkpoint = torch.load(LATEST_MODEL_CHECKPOINT,weights_only=True)
    log_reg.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch_start = checkpoint['epoch']+1
    best_val_acc = checkpoint['best_val_acc']
for ep in range(epoch_start,NUM_EPOCHS):
    ep_loss,tot_items = 0,0
    log_reg.train()
    for x,y in train_loader:
        x,y = x.to(DEVICE),y.to(DEVICE)
        y_pred = log_reg(x)
        loss = criterion(y_pred,y)
        ep_loss += loss.item()*x.size(0)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        tot_items += x.size(0)
    train_loss.append(ep_loss/tot_items)
    torch.save({
            'epoch':ep,
            'model_state_dict':log_reg.state_dict(),
            'optimizer_state_dict':optimizer.state_dict(),
            'best_val_acc':best_val_acc
            },LATEST_MODEL_CHECKPOINT)
    #Validation
    tot_val_items,val_ep_loss,val_crct = 0,0,0
    log_reg.eval()
    with torch.no_grad():
        for x,y in val_loader:
            x,y = x.to(DEVICE),y.to(DEVICE)
            y_pred = log_reg(x)
            val_crct += (y==torch.argmax(y_pred,dim=1)).sum().item()
            loss = criterion(y_pred,y)
            val_ep_loss += loss.item()*x.size(0)
            tot_val_items += x.size(0)
        if val_crct/tot_val_items > best_val_acc:
            torch.save({
            'epoch':ep,
            'model_state_dict':log_reg.state_dict(),
            'optimizer_state_dict':optimizer.state_dict(),
            'val_acc': val_crct/tot_val_items,
            'val_loss': val_ep_loss/tot_val_items
            },BEST_MODEL_CHECKPOINT)
            best_val_acc = val_crct/tot_val_items
        val_loss.append(val_ep_loss/tot_val_items)
    if ep%10 == 0: print(f"Epoch {ep} : Train_Loss = {round(train_loss[-1],PRECISION)}, Val_Loss = {round(val_loss[-1],PRECISION)}, Val_Acc = {round(val_crct/tot_val_items,PRECISION)}")

# Plotting     
plt.figure(figsize=(8, 6))
plt.plot(range(len(train_loss)), train_loss, label="Train Loss")
plt.plot(range(len(val_loss)), val_loss, label="Validation Loss")
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig(LOSS_CURVE_SAVE_PATH, dpi=300)
plt.close()

# Testing
test_crct, tot_items = 0, 0
log_reg = nn.Sequential(nn.Flatten(),nn.Linear(IN_FEATURES,OUT_FEATURES))
log_reg = log_reg.to(DEVICE)
log_reg.load_state_dict(torch.load(BEST_MODEL_CHECKPOINT,weights_only=False)['model_state_dict'])
with torch.no_grad():
    for x,y in test_loader:
        x,y = x.to(DEVICE),y.to(DEVICE)
        y_pred = torch.argmax(log_reg(x),dim=1)
        test_crct += (y==y_pred).sum().item()
        tot_items += x.size(0)
print(f"Test Accuracy : {round(test_crct/tot_items,PRECISION)}")