import torch
from tqdm import tqdm

'''
Train the model on loader using criterion and optimizer on device
Works only for CE!!
Supervised Training Assumption
'''
def fit(model,loader,criterion,optimizer,device,clip=None):
    model.train()
    loss_val, total_size, correct = 0, 0, 0
    for x,y in tqdm(loader,leave=False):
        x,y = x.to(device,non_blocking=True), y.to(device,non_blocking=True).long()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits,y)
        loss.backward()
        if clip: torch.nn.utils.clip_grad_norm_(model.parameters(),clip)
        optimizer.step()
        loss_val += loss.item() * y.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total_size += y.size(0)
    return loss_val/total_size, correct/total_size

'''
Test the model on loader on device, criterion for test_loss
Works only for CE!!
Supervised Assumption
'''
@torch.no_grad()
def evaluate(model,loader,criterion,device):
    model.eval()
    loss_val, total_size, correct = 0, 0, 0
    for x,y in tqdm(loader,leave=False):
        x,y = x.to(device,non_blocking=True), y.to(device,non_blocking=True).long()
        logits = model(x)
        loss = criterion(logits,y)
        loss_val += loss.item() * y.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total_size += y.size(0)
    return loss_val/total_size, correct/total_size