import os
import time

import torch
import torch.nn as nn
from torch.nn import functional as F

from bpe import BytePairTokenizer

torch.set_float32_matmul_precision("high")

# hyperparameters
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200

n_embd = 384 #this is the number of token embedding dimensions.
n_head = 6 # This is the number of attention heads in each block
n_layer = 6 #No. of blocks
dropout = 0.2

total_batch_size = 524288  # total tokens per step (target) // this could be between the 0.1% to 1% of the total tokens size of the dataset.
B = batch_size  # your micro-batch size
T = block_size

grad_accum_steps = total_batch_size // (B * T)
assert grad_accum_steps >= 1


# tokenization and train / val data split
torch.manual_seed(1337)

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()


tokenizer = BytePairTokenizer(vocab_size=1000)

if not os.path.exists("tokenizer.pkl"):
    tokenizer.train(text)
    tokenizer.save("tokenizer.pkl")
else:
    tokenizer.load("tokenizer.pkl")

data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
vocab_size = len(tokenizer.vocab)


n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


class DataLoaderLite:
    def __init__(self, B, T, data):
        self.B = B
        self.T = T
        self.data = data
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T

        buf = self.data[self.current_position : self.current_position + B * T + 1]

        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)

        self.current_position += B * T

        if self.current_position + B * T + 1 > len(self.data):
            self.current_position = 0

        return x.to(device), y.to(device)


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_head = n_head
        self.head_size = n_embd // n_head

        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        B, T, C = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)

        # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2) #Query is something that the token is looking for
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2) # The key is the one that tells the rest of the sequence: This is what I represent, and this is why you might went to pay to me.
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2) # Value is something that the token will give when it finds a good match for itself.

        # the dot product of query and key gives out the compatiblity score.
        # TTo prevent the gradients from exploding, we scale by the square root of the dimension dk. The dk means embedding size / no. of attention heads
        # Then softmax is applied. 
        # flash attention
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=dropout if self.training else 0.0
        )
        # back to (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.proj(out) #the output is context aware tokens.


class FeedForward(nn.Module): #here each token starts thinking independently based on the output of the self-attention block to decide what to do with that information.
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)


# Transformer block
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention()
        self.ln2 = nn.LayerNorm(n_embd)
        self.ff = FeedForward()

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


# GPT Model
class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = (
            self.token_emb.weight
        )  # Reduces paramters + improves generalisation

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.token_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=device))

        x = tok + pos
        x = self.blocks(x)
        x = self.ln_f(x)

        logits = self.lm_head(x) #here we get logits. These are the raw scores that the model has calculated.

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx


# Training loop
use_compile = True
model = GPT().to(device)
if use_compile:
    model = torch.compile(model)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
train_loader = DataLoaderLite(B, T, train_data)
val_loader = DataLoaderLite(B, T, val_data)


@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    loaders = {"train": train_loader, "val": val_loader}
    for split, loader in loaders.items():
        losses = torch.zeros(eval_iters)
        loader.reset()

        for k in range(eval_iters):
            X, Y = loader.next_batch()
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                _, loss = model(X, Y)
                losses[k] = loss.item()
                out[split] = losses.mean()
    return out


for iter in range(max_iters):
    t0 = time.time()

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"{iter}: train {losses['train']:.4f}, val {losses['val']:.4f}")

    optimizer.zero_grad()
    loss_accum = 0.0

    for micro_step in range(grad_accum_steps):
        xb, yb = train_loader.next_batch()

        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(xb, yb)

        loss = loss / grad_accum_steps
        loss_accum += loss.detach()

        loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    t1 = time.time()
    dt = t1 - t0

    tokens_processed = B * T * grad_accum_steps
    tokens_per_sec = tokens_processed / dt

    print(f"step {iter} | loss {loss_accum.item():.4f} | {tokens_per_sec:.2f} tok/s")
