import torch
import torch.nn as nn

from rnn_utils import N_LETTERS
from rnn_utils import load_data,random_training_example,name_to_tensors
import matplotlib.pyplot as plt

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, input_tensor, hidden_tensor):
        combined = torch.cat((input_tensor, hidden_tensor), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden
    
    def init_hidden(self): return torch.zeros(1, self.hidden_size)

data = load_data()
countries = sorted(list(data.keys()))
num_countries = len(countries)
hidden_size = 128
rnn = RNN(N_LETTERS,hidden_size,num_countries)

def country_from_output(output): return countries[torch.argmax(output).item()]

criterion = nn.NLLLoss()
learning_rate = 0.005
optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)

def train(line_tensor, category_tensor):
    hidden = rnn.init_hidden()
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
    loss = criterion(output, category_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return output, loss.item()

current_loss = 0
all_losses = []
plot_steps, print_steps = 1000, 5000
n_iters = 100000

for i in range(n_iters):
    country,name,country_tensor,name_tensors = random_training_example(data)
    output, loss = train(name_tensors, country_tensor)
    current_loss += loss 
    if (i+1) % plot_steps == 0:
        all_losses.append(current_loss / plot_steps)
        current_loss = 0
    if (i+1) % print_steps == 0:
        guess = country_from_output(output)
        correct = "CORRECT" if guess == country else f"WRONG ({country})"
        print(f"{i+1} {(i+1)/n_iters*100} {loss:.4f} {name} / {guess} {correct}")

plt.figure()
plt.plot(all_losses)
plt.savefig("results/plots/rnn_loss.jpg")

def predict(input_line):
    print(f"\n> {input_line}")
    with torch.no_grad():
        line_tensor = name_to_tensors(input_line)
        hidden = rnn.init_hidden()
        for i in range(line_tensor.size()[0]):
            output, hidden = rnn(line_tensor[i], hidden)
        guess = country_from_output(output)
        print(guess)

while True:
    sentence = input("Input:")
    if sentence == "quit":
        break
    predict(sentence)