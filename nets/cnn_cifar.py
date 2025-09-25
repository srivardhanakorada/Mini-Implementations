## Imports
import torch
from torchvision import transforms
from torch.utils.data import random_split, DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter

## Setup
SEED = 42
BATCH_SIZE = 128
IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32
IMAGE_CHANNELS = 3
NUM_CLASSES = 10
NUM_EPOCHS = 10
DEVICE = "cuda:0"
LR = 1e-3
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed(SEED)

## Data
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
class CIFAR_Dataset(Dataset):
    def __init__(self,path,transform,num_splits=1):
        batches = [unpickle(f"{path}/data_batch_{i}") for i in range(1,num_splits+1)]
        x = [batches[i][b'data'] for i in range(0,len(batches))]
        y = [batches[i][b'labels'] for i in range(0,len(batches))]
        self.x = np.vstack(np.asarray(x))
        self.x = self.x.reshape(-1,IMAGE_CHANNELS,IMAGE_WIDTH,IMAGE_HEIGHT)
        self.x = np.transpose(self.x,(0,2,3,1))
        self.y = np.hstack(np.asarray(y))
        self.transform = transform
    def __len__(self): return self.x.shape[0]
    def __getitem__(self, idx):
        img = self.x[idx]
        label = self.y[idx]
        if self.transform: img = self.transform(img)
        return img,label
tfm = transforms.Compose([transforms.ToTensor()])
train_full_data = CIFAR_Dataset(path="data/cifar-10-batches-py/train",transform=tfm,num_splits=5)
test_data = CIFAR_Dataset(path="data/cifar-10-batches-py/test",transform=tfm,num_splits=1)
train,val = random_split(train_full_data,[0.8,0.2],generator=torch.Generator().manual_seed(SEED))
train_loader = DataLoader(train,batch_size=BATCH_SIZE,shuffle=True,pin_memory=True,num_workers=4)
val_loader = DataLoader(val,batch_size=BATCH_SIZE,shuffle=False,pin_memory=True,num_workers=4)
test_loader = DataLoader(test_data,batch_size=BATCH_SIZE,shuffle=False,pin_memory=True,num_workers=4)

## Model
class CNN(nn.Module):
    def __init__(self,image_channels):
        super().__init__()
        self.conv_net_one = nn.Sequential(
            nn.Conv2d(image_channels,image_channels*128,kernel_size=3,stride=3,padding=2),# 384 * 12 * 12
            nn.BatchNorm2d(image_channels*128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.AvgPool2d(kernel_size=2,stride=1),  # 384 * 11 * 11
        )
        self.conv_net_two = nn.Sequential(
            nn.Conv2d(image_channels*128,image_channels*128,kernel_size=3,stride=1,padding=1), # 384 * 11 * 11
            nn.BatchNorm2d(image_channels*128),
            nn.ReLU(),
        )
        self.linear_net = nn.Linear(384*11*11,NUM_CLASSES)
    def forward(self,x):
        x =  self.conv_net_one(x)
        y = torch.add(x,self.conv_net_two(x))
        z = y.view(y.size(0),-1)
        z = self.linear_net(z)
        return z
    
## Training
model = CNN(IMAGE_CHANNELS).to(device=DEVICE)
optimizer = optim.Adam(model.parameters(),lr=LR)
criterion = nn.CrossEntropyLoss()
train_losses,val_losses = [],[]
writer = SummaryWriter("runs/mnist_cifar")
for ep in range(NUM_EPOCHS):
    running_loss, total_size, val_loss,val_acc, val_size = 0,0,0,0,0
    model.train()
    for x,y in train_loader:
        x,y = x.to(DEVICE),y.to(DEVICE)
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

## Plotting
plt.figure(figsize=(10,8))
plt.title("Losses")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.plot(list(range(len(train_losses))),train_losses,label="Train")
plt.plot(list(range(len(val_losses))),val_losses,label="Val")
plt.legend()
plt.savefig("results/plots/cnn_cifar_losses.png")

## Inference
test_acc, total_size = 0, 0
model.eval()
with torch.no_grad():
    for x,y in test_loader:
        x,y = x.to(DEVICE),y.to(DEVICE)
        y_pred = model(x)
        test_acc += (y == torch.argmax(y_pred,dim=1)).sum().item()
        total_size += x.size(0)
    print(f'Test Acc : {round(test_acc/total_size,4)*100}%')