## Imports
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
import evaluate
from accelerate import Accelerator

## Parameters
checkpoint = "bert-base-cased"
batch_size = 8
DEVICE = 'cuda'
LR = 5e-5
NUM_EPOCHS= 10
accelerator = Accelerator()

## Dataset
raw_dataset = load_dataset("glue","mrpc")
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
def tokenize_function(x): return tokenizer(x["sentence1"],x["sentence2"],truncation=True) 
tokenized_dataset = raw_dataset.map(tokenize_function,batched=True)
tokenized_dataset = tokenized_dataset.remove_columns(["sentence1", "sentence2", "idx"]).rename_column("label","labels")
tokenized_dataset.set_format("torch")
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
train_dataloader = DataLoader(tokenized_dataset["train"], shuffle=True, batch_size=batch_size, collate_fn=data_collator)
val_dataloader = DataLoader(tokenized_dataset["validation"], shuffle=True, batch_size=batch_size, collate_fn=data_collator)


model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
model.to(DEVICE)

metric = evaluate.load('glue', 'mrpc')
optimizer = AdamW(model.parameters(), lr = LR)
scheduler = get_scheduler('linear', optimizer = optimizer, num_warmup_steps=0, num_training_steps=NUM_EPOCHS*len(train_dataloader))

model,optimizer,train_dataloader,val_dataloader = accelerator.prepare(model,optimizer,train_dataloader,val_dataloader)

progress_bar = tqdm(range(NUM_EPOCHS*len(train_dataloader)))
for epoch in range(NUM_EPOCHS):
    for batch in train_dataloader:
        batch = {k:v.to(DEVICE) for k,v in batch.items()}
        output = model(**batch)
        loss = output.loss
        accelerator.backward(loss)
        optimizer.step()        
        scheduler.step()
        optimizer.zero_grad()   
        progress_bar.update(1)

    model.eval()

    for batch in val_dataloader:
        batch = {k:v.to(DEVICE) for k,v in batch.items()}
        with torch.no_grad():
            output = model(**batch)
        predictions = torch.argmax(output.logits, dim = -1)
        metric.add_batch(predictions=predictions, references=batch['labels'])
    
    print(f'epoch : {epoch}, {metric.compute()}')            
