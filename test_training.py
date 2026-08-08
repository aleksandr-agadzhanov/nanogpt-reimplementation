import time

import torch

from data_loader import DataLoader
from train_gpt2 import GPT, GPTConfig

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.mps.is_available():
    device = "mps"
print(f"Using device - {device}")

train_loader = DataLoader("tiny_shakespeare.txt", 8, 1024)

model = GPT(GPTConfig())
model.to(device)

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)
if torch.mps.is_available():
    torch.mps.manual_seed(1337)

# torch.set_float32_matmul_precision("high")

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for i in range(50):
    t0 = time.time()
    x, y = train_loader.get_next_batch()
    x = x.to(device)
    y = y.to(device)
    optimizer.zero_grad()
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    # torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0) * 1000 # ms
    tokens_per_second = (train_loader.batch_size * train_loader.context_size) / (t1 - t0)
    print(f"Step - {i}, loss - {loss.item()}, dt - {dt:.2f}ms, tokens/s - {tokens_per_second:.2f}")

print(loss)
