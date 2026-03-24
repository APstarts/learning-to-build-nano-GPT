import torch
from bigram import BigramLanguageModel

# device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# load checkpoint
checkpoint = torch.load('audit_model.pt', map_location=device)

vocab_size = checkpoint['config']['vocab_size']

# rebuild model
model = BigramLanguageModel(vocab_size=vocab_size)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# load vocab
stoi = checkpoint['stoi']
itos = checkpoint['itos']

# encoder / decoder
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# prompt
prompt = """### Instruction:
Write an internal audit observation.

### Input:
Area: Inventory
Process: Stock Reconciliation
Condition: Stock mismatch observed.
Criteria: Monthly reconciliation required.
Cause: No periodic checks.
Impact: Risk of misstatement.

### Response:
"""

# convert to tensor
context = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)

# generate
output = model.generate(context, max_new_tokens=300)

# decode
text = decode(output[0].tolist())

# stop at END token
if "### END" in text:
    text = text.split("### END")[0]

print(text)