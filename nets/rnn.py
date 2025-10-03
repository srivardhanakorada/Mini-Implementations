import torch 
import torch.nn as nn

from utils.rnn_utils import load_data, random_training_example, ALL_LETTERS, N_LETTERS, letter_to_tensor, line_to_tensor

class RNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(RNN, self).__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.i2h = nn.Linear(in_dim + hidden_dim, hidden_dim)
        self.i2o = nn.Linear(in_dim + hidden_dim, out_dim)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_, hidden_):
        combined = torch.cat((input_, hidden_), 1)

        hidden = self.i2h(combined)
        output = self.i2o(combined)

        output = self.softmax(output)

        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_dim)
    

catogery_lines , all_catogeries = load_data()
out_dim = len(all_catogeries)
hidden_dim = 128



rnn = RNN(N_LETTERS, hidden_dim, out_dim)

# one step
input_tensor = letter_to_tensor('A')
hidden_tensor = rnn.init_hidden()

output, next_hidden = rnn(input_tensor, hidden_tensor)
print(output.size())
print(next_hidden.size())

# whole sequence/name
input_tensor = line_to_tensor('Albert')
hidden_tensor = rnn.init_hidden()

output, next_hidden = rnn(input_tensor[0], hidden_tensor)
print(output.size())
print(next_hidden.size())


def catogery_from_output(output):
    catogery = torch.argmax(output).item()
    return all_catogeries[catogery]

criterion = nn.NLLLoss()
lr = 0.005
optimizer = torch.optim.SGD(params=rnn.parameters(), lr=lr)

def train(line_tensor, catogery_tensor):
    hidden = rnn.init_hidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
    
    loss = criterion(output, catogery_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return output, loss.item()

current_loss = 0

n_iters = 100000

breakpoint()
for i in range(n_iters):
    catogery, line, catogery_tensor, line_tensor = random_training_example(catogery_lines, all_catogeries)

    output, loss = train(line_tensor, catogery_tensor)
    current_loss += loss

    if (i+1) % 5000 == 0:
        guess = catogery_from_output(output)
        correct = "CORRECT" if guess == catogery else f"WRONG ({catogery})"
        print(f"{i+1} {(i+1)/n_iters*100} {loss:.4f} {line} / {guess} {correct}")



def predict(input_line):
    print(f"\n> {input_line}")
    with torch.no_grad():
        line_tensor = line_to_tensor(input_line)
        
        hidden = rnn.init_hidden()
    
        for i in range(line_tensor.size()[0]):
            output, hidden = rnn(line_tensor[i], hidden)
        
        guess = catogery_from_output(output)
        print(guess)


while True:
    sentence = input("Input:")
    if sentence == "quit":
        break
    
    predict(sentence)