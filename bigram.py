import torch
import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)


with open("austen.txt", 'r', encoding='utf-8') as f1:
    text = f1.read()

# hyperparameters
block_size = 8     # the maximum size of the block that is fed to the transformer at once
batch_size = 32    # how many independent sequences will we process in parallel?

chars = sorted(list(set(text)))
vocab_size = len(chars)

# create a mapping from string to integers
stoi = {chars[i]:i for i in range(vocab_size)}
itos = {i:chars[i] for i in range(vocab_size)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda i: "".join([itos[j] for j in i])

# encode the entire dataset and store it in a tensor
data = torch.tensor(encode(text), dtype=torch.long)

# split up data into train and validation sets
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    # generate a small batch of data of inputs x and target y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])

    return x, y

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B, T) tensor of integers
        logits = self.token_embedding_table(idx) # (B, T, C) Batch, Time, Channel (vocab size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the prediction
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)

        return idx

# get training data in batches
xb, yb = get_batch('train')

m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)

# create a Pytorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

for steps in range(10000):
    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(loss.item())


print(decode(m.generate(idx=torch.zeros([1,1], dtype=torch.long), max_new_tokens=100)[0].tolist()))

