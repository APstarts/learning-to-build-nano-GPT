# GPT From Scratch (NanoGPT-style) — Detailed README

This repository implements a **GPT-style Transformer model from scratch**, heavily inspired by :contentReference[oaicite:0]{index=0}’s teaching philosophy:
> Build everything step-by-step, understand every tensor, and remove abstraction layers.

This README is not just usage documentation — it is a **deep technical walkthrough** of:
- Every component
- Every important line of code
- Underlying data structures
- Mathematical and conceptual foundations

---

## 1. High-Level Architecture

At a high level, this code implements:
Raw Text → Tokenizer → Integer Tokens → Transformer → Next Token Prediction

Key components:
- Byte Pair Encoding tokenizer
- Data batching system
- Transformer (Attention + FeedForward)
- Training loop with gradient accumulation

---

## 2. Hyperparameters (Design Decisions)

```python
batch_size = 64
block_size = 256
```

Explanation
batch_size (B) → number of sequences processed in parallel
block_size (T) → sequence length (context window)
Data Structure

Input tensor shape:

(B, T) → (64, 256)

```
n_embd = 384
n_head = 6
n_layer = 6
```
Explanation
n_embd → embedding dimension (vector size per token)
n_head → number of attention heads
n_layer → number of transformer blocks
Constraint
head_size = n_embd / n_head = 384 / 6 = 64

Each head operates on a 64-dimensional subspace.

```
total_batch_size = 524288
grad_accum_steps = total_batch_size // (B * T)
```

Concept (Karpathy Insight)

Instead of increasing GPU memory:
> Simulate large batch training via gradient accumulation

## 3. Tokenization (Byte Pair Encoding)

```
tokenizer = BytePairTokenizer(vocab_size=1000)
```
Concept
Converts raw text → integers
Learns most frequent subwords

Example:

"playing" → ["play", "ing"]

Output Data Structure
data = torch.tensor([...], dtype=torch.long)

Shape:

(Num_Tokens,)

##4. Train / Validation Split
```
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
```
**Concept**
90% training
10% validation

## 5. DataLoaderLite (Critical for Understanding GPT)
**Why +1?**
Because:
Input (x):  [t1, t2, t3]
Target (y): [t2, t3, t4]
We need **shifted labels**.

```
x = buf[:-1].view(B, T)
y = buf[1:].view(B, T)
```

**Concept**
Ths creates
x: model input
y: expected output (next token)

## 6. Multi-Head Attention (CORE OF TRANSFORMERS)
```
self.qkv = nn.Linear(n_embd, 3 * n_embd)
```

Concept

Single matrix computes:

Query (Q)
Key (K)
Value (V)

Efficient trick:

Instead of 3 layers → use 1 layer and split
```
q, k, v = qkv.split(C, dim=2)
```

Data Structure

(B, T, 3C) → split → 3 × (B, T, C)

```
q = q.view(B, T, n_head, head_size).transpose(1, 2)
```

Shape Transformation
(B, T, C) → (B, n_head, T, head_size)

**Why?**

Each head processes independently.

**Attention Formula (Core Idea)**

```
out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

**Mathematical Concept**

Attention:

Attention(Q, K, V) = softmax(QKᵀ / √d_k) V

**Karpathy Explanation**
Query → "What am I looking for?"
Key → "What do I contain?"
Value → "What do I give?"

``` is_casual=True```
Prevents cheating:
Token t cannot see future tokens

## 7. FeedForward Network
```nn.Linear(n_embd, 4 * n_embd)```
**Concept**

Expand → compress

384 → 1536 → 384

**Why?**

Adds non-linearity and capacity

```nn.GELU()```

**Activation**

Smooth alternative to ReLU:

- Better for transformers

## 8. Transformer Block
```x = x + self.attn(self.ln1(x))```

**Concepts Combined**
- LayerNorm
- Residual connection

**Residual Connection (Critical Insight)**
output = input + transformation(input)

**Why?**

- Prevents vanishing gradients
- Enables deep networks

## 9. GPT Model
``self.token_emb = nn.Embedding(vocab_size, n_embd)``

**Concept**

Lookup table:

token_id → vector

```self.pos_emb = nn.Embedding(block_size, n_embd)```

**Why?**

Transformers have no sense of order.

So we add:

position encoding
```x = tok + pos```

**Concept**

Combine:
- What the token is
- Where it is

```self.lm_head.weight = self.token_emb.weight```

**Weight Tying (Karpathy Insight)**
- Reduces parameters
- Improves generalization

## 10. Loss Function
```loss = F.cross_entropy(logits.view(-1, vocab), targets.view(-1))```

**Explanation**

Flatten:

(B, T, vocab) → (B*T, vocab)

**Concept**

Predict next token probability.

## 11. Text Generation
```probs = F.softmax(logits, dim=-1)
next_token = torch.multinomial(probs, 1)
```

**Concept**
- Convert logits → probabilities
- Sample (not argmax)

**Why sampling?**
→ More creative text generation

## 12. Training Loop
```loss.backward()```

**Concept**

Backpropagation:
- Compute gradients
```torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)```

**Why?**

Prevent exploding gradients.

```optimizer.step()```

**Concept**

Update weights using AdamW:
- Adaptive learning rate
- Weight decay regularization

## 13. Mixed Precision Training
with torch.autocast(dtype=torch.bfloat16):

**Concept**
- Faster computation
- Less memory usage

## 14. Key Karpathy Principles Applied
**1. Build Everything Yourself**
No HuggingFace abstractions.

**2. Understand Shapes**

Every operation is about tensor shapes:

(B, T, C)

**3. Attention = Communication**

Tokens "talk" to each other.

**4. Scaling Matters**
Gradient accumulation
Mixed precision
Efficient batching

## 15. File Reference

Full implementation:


## 16. Mental Model (Most Important)

Think of GPT as:

A system where:
Each token asks:
→ "Who should I pay attention to?"
→ "What information should I extract?"
→ "What should I output next?"

## 17. If You Truly Understand This

- You should now be able to:
- Modify architecture (heads, layers)
- Implement your own tokenizer
- Debug tensor shape errors
- Extend to:
    - Classification
    - Information extraction
    - Summarization

## 18. Suggested Next Step (For You)

Given your goal (information extraction from concalls):

Add:
- Classification head
- Span prediction
- Instruction tuning