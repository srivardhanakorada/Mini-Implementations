import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

## Data Creation
torch.manual_seed(0)
N = 1024
x = torch.rand(N,1)*4 - 2
noise = torch.randn(N,1)*0.1
y = 3.0 * x + 0.5 + noise

## Train/val split
reordered_idxs = torch.randperm(N)
train_split,val_split = reordered_idxs[:800], reordered_idxs[800:]
x_train,y_train = x[train_split],y[train_split]
x_val,y_val = x[val_split],y[val_split]

## Linear Model
linear_reg = nn.Sequential(nn.Linear(1,1))

## Train & Eval Loop
criterion = nn.MSELoss()
optimizer = optim.SGD(linear_reg.parameters(),lr=1e-2)
num_epochs = 200
train_losses,val_losses = [], []
for ep in range(num_epochs):
    linear_reg.train()
    optimizer.zero_grad(set_to_none=True)
    pred_val = linear_reg(x_train)
    loss = criterion(pred_val,y_train)
    loss.backward()
    optimizer.step()
    linear_reg.eval()
    with torch.no_grad():
        val_pred = linear_reg(x_val)
        val_loss = criterion(val_pred, y_val)
    if (ep+1)%10 == 0:
        print(f'Epoch {ep+1} ||  Train : {round(loss.item()/len(x_train),3)}, Test : {round(val_loss.item()/len(x_val),3)}')
        train_losses.append(loss.item()/len(x_train))
        val_losses.append(val_loss.item()/len(x_val))

## Print Weights
w = linear_reg[0].weight.item()
b = linear_reg[0].bias.item()
print(f"Learned w={w:.4f}, b={b:.4f}")

## Plotting
plt.figure()
plt.plot(list(range(len(train_losses))),train_losses,label="train_mse")
plt.plot(list(range(len(val_losses))),val_losses,label="val_mse")
plt.legend()
plt.title("Linear Regression Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("results/linear_reg.png")