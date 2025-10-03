## Imports
import torch
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import random_split, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter

## Setup
SEED = 42
BATCH_SIZE = 128
IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28
IMAGE_CHANNELS = 1
NUM_CLASSES = 10
NUM_EPOCHS = 10
DEVICE = "cuda:0"
NUM_LAYERS = 2
HIDDEN_SIZE = 128
LR = 1e-3
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(SEED)

## Data
tfm = transforms.Compose([transforms.ToTensor()])
train_full_data = MNIST(root="data/MNIST",train=True,transform=tfm,download=True)
train,val = random_split(train_full_data,[0.8,0.2],generator=torch.Generator().manual_seed(SEED))
train_loader = DataLoader(train,batch_size=BATCH_SIZE,shuffle=True,pin_memory=True,num_workers=4)
val_loader = DataLoader(val,batch_size=BATCH_SIZE,shuffle=False,pin_memory=True,num_workers=4)
test_data = MNIST(root="data/MNIST",train=False,transform=tfm,download=True)
test_loader = DataLoader(test_data,batch_size=BATCH_SIZE,shuffle=False,pin_memory=True,num_workers=4)

## Model
class RNN(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,output_size):
        super().__init__()
        self.rnn = nn.RNN(input_size,hidden_size,num_layers,batch_first=True)
        self.linear_layer = nn.Linear(hidden_size,output_size)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
    def forward(self,x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        _,h_T = self.rnn(x,h0)
        return self.linear_layer(h_T[-1])

## Training
model = RNN(IMAGE_WIDTH,HIDDEN_SIZE,NUM_LAYERS,NUM_CLASSES).to(DEVICE)

optimizer = optim.Adam(model.parameters(),lr=LR)
criterion = nn.CrossEntropyLoss()
train_losses,val_losses = [],[]
writer = SummaryWriter("runs/mnist_rnn")
for ep in range(NUM_EPOCHS):
    running_loss, total_size, val_loss,val_acc, val_size = 0,0,0,0,0
    model.train()
    for x,y in train_loader:
        x,y = x.to(DEVICE),y.to(DEVICE)
        x = x.squeeze(1)
        y_pred = model(x)
        loss = criterion(y_pred,y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()*x.size(0)
        total_size += x.size(0)
    model.eval()
    with torch.no_grad():
        for x,y in val_loader:
            x,y = x.to(DEVICE),y.to(DEVICE)
            x = x.squeeze(1)
            y_pred = model(x)
            loss = criterion(y_pred,y)
            val_loss += loss.item()*x.size(0)
            val_size += x.size(0)
            val_acc += (y == torch.argmax(y_pred,dim=1)).sum().item()
    print(f"Epoch {ep} : Train_Loss = {running_loss/total_size}, Val_Loss = {val_loss/val_size}, Val_Acc = {val_acc/val_size}")
    writer.add_scalars(
        "loss",
        {"train": running_loss/total_size, "val": val_loss/val_size},
        ep,
    )
    writer.add_scalar("val_acc",val_acc/val_size,ep)
    train_losses.append(running_loss/total_size)
    val_losses.append(val_loss/val_size)
writer.close()

plt.figure(figsize=(10,8))
plt.title("Losses")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.plot(list(range(len(train_losses))),train_losses,label="Train")
plt.plot(list(range(len(val_losses))),val_losses,label="Val")
plt.legend()
plt.savefig("results/plots/ffn_losses.png")

test_acc, total_size = 0, 0
model.eval()
with torch.no_grad():
    for x,y in test_loader:
        x = x.view(x.size(0), IMAGE_HEIGHT, IMAGE_WIDTH)
        x,y = x.to(DEVICE),y.to(DEVICE)
        y_pred = model(x)
        test_acc += (y == torch.argmax(y_pred,dim=1)).sum().item()
        total_size += x.size(0)
    print(f'Test Acc : {100.0 * test_acc/total_size:.2f}%')