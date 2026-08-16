import math
import os
import time
from contextlib import nullcontext
import tiktoken

import torch
from torch import distributed, nn
from torch.nn.parallel import DistributedDataParallel as DDP

from data_loader import DataLoader
from GPT import GPT, GPTConfig
from hellaswag import iterate_examples, render_example, get_most_likely_row

# torchrun --standalone --nproc_per_node=8 train_gpt2.py

# DistributedDataParallel
ddp_is_enabled = "RANK" in os.environ
if ddp_is_enabled:
    distributed.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.mps.is_available():
        device = "mps"
print(f"Using device - {device}")

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)
if torch.mps.is_available():
    torch.mps.manual_seed(1337)

model = GPT(GPTConfig())
model.to(device)

# Optimization 3 TODO
use_compile = False # compilation interferes with eval and generation
if use_compile:
    model = torch.compile(model)

if ddp_is_enabled:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp_is_enabled else model

# Could run without gradient accumulation if batch_size * sequence_size * ddp_world_size = total_batch_size
total_batch_size = 1024 # 524288
batch_size = 16  # for GPT-2 use 64, for GPT-3 use 32
sequence_size = 64 # GPT-2 uses 1024, GPT-3 uses 2048
num_gradient_accumulation_steps = total_batch_size // (batch_size * sequence_size * ddp_world_size)
if master_process:
    print(f"Number of gradient accumulation steps = {total_batch_size} // ({batch_size} * {sequence_size} * {ddp_world_size}) = {num_gradient_accumulation_steps}")

train_loader = DataLoader("tiny_shakespeare.txt", 4, 32, ddp_rank, ddp_world_size, "train")
val_loader = DataLoader("tiny_shakespeare.txt", 4, 32, ddp_rank, ddp_world_size, "val")

# Optimization 1
torch.set_float32_matmul_precision("high")

max_lr = 6e-4 # could 3x this
min_lr = 0.1 * max_lr
warmup_steps = 10   # 715 = 375e6 / 2**19
max_steps = 50   # 19073 = 2**19 / 10e9
def get_lr(step):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    elif step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coefficient = 0.5 * (1 + math.cos(math.pi * decay_ratio))
    return min_lr + coefficient * (max_lr - min_lr)

optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)
tokenizer = tiktoken.get_encoding("gpt2")

log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "log.txt")
with open(log_file, 'w') as f:
    pass

for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    # Validation
    if step % 250 == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accumulated = 0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.get_next_batch()
                x = x.to(device)
                y = y.to(device)
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accumulated = val_loss_accumulated + loss.detach()
        if ddp_is_enabled:
            distributed.all_reduce(val_loss_accumulated, op=distributed.ReduceOp.AVG)
        if master_process:
            print(f"Validation loss {val_loss_accumulated.item():.4f}")
            with open(log_file, 'w') as file:
                file.write(f"{step} val {val_loss_accumulated.item():.4f}\n")
            if step > 0 and (step % 5000 == 0 or last_step):
                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                checkpoint = {
                    "model": model.state_dict(),
                    "config": raw_model.config,
                    "step": step,
                    "val_loss": val_loss_accumulated
                }
                torch.save(checkpoint, checkpoint_path)

    # Hellaswag evaluation
    if (step % 250 == 0 or last_step) and not use_compile:
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_examples("val")):
            if i % ddp_world_size != ddp_rank:
                continue
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)
            with torch.no_grad():
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(tokens)
                _, pred_norm, _ = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)
            if ddp_is_enabled:
                num_total = torch.tensor(num_total, dtype=torch.long, device=device)
                num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
                distributed.all_reduce(num_total, op=distributed.ReduceOp.SUM)
                distributed.all_reduce(num_correct_norm, op=distributed.ReduceOp.SUM)
                num_total = num_total.item()
                num_correct_norm = num_correct_norm.item()
            acc_norm = num_correct_norm / num_total
            if master_process:
                print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:4f}")
                with open(log_file, 'w') as file:
                    file.write(f"{step} hella {acc_norm:4f}\n")

    # Generation - REQUIRES DISABLING MODEL COMPLILE, SO DON'T DO IT TO TRAIN (test first, maybe not)
    if ((step > 0 and step % 250 == 0) or last_step) and not use_compile:
        model.eval()
        num_return_sequences = 4
        max_length = 32
        tokens = tokenizer.encode("Hello, I'm a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank)
        while xgen.size(1) < max_length:
            with torch.no_grad():
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(xgen, last_position_only=True)
                logits = logits[:, -1, :]
                probs = logits.softmax(-1)
                topk_probs, topk_indices = torch.topk(probs, 50, -1)
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
                xcol = torch.gather(topk_indices, -1, ix)
                x = torch.cat((x, xcol), dim=-1)
        for i in range(num_return_sequences):
            tokens = x[i, :max_length].tolist()
            decoded = tokenizer.decode(tokens)
            print(f"Rank {ddp_rank}, sample {i + 1}: {decoded}")

    optimizer.zero_grad()

    loss_accumulated = 0.0
    # Optimization 2
    for micro_step in range(num_gradient_accumulation_steps):
        x, y = train_loader.get_next_batch()
        x = x.to(device)
        y = y.to(device)
        is_last_micro_step = micro_step == num_gradient_accumulation_steps - 1

        if ddp_is_enabled:
            model.require_backward_grad_sync = is_last_micro_step

        # Only sync gradients across processes on the last micro-step
        sync_context = model.no_sync() if ddp_is_enabled and not is_last_micro_step else nullcontext()
        with sync_context:
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            # Below is needed to account for gradient accumulation, so that we don't simply sum the gradients but take their mean
            loss = loss / num_gradient_accumulation_steps
            loss_accumulated = loss_accumulated + loss
            loss.backward()

    # To synchronize the loss across all processed
    if ddp_is_enabled:
        distributed.all_reduce(loss_accumulated, op=distributed.ReduceOp.AVG)
    
    # Clipping the gradient l2-norm to 1 so that if we get unlucky with a batch, the model shock coming from high gradients is avoided
    norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    optimizer.step()
    # torch.cuda.synchronize() # wait for the GPU to finish work
    t1 = time.time()
    dt = (t1 - t0) * 1000 # ms
    tokens_processed = train_loader.batch_size * train_loader.context_size * num_gradient_accumulation_steps * ddp_world_size
    tokens_per_second = tokens_processed / (t1 - t0)

    if master_process:
        print(f"Step {step:4d} | loss {loss_accumulated.item():.6f} | lr {lr:.4e} | norm {norm:.4f} | dt {dt:.2f}ms | tokens/s {tokens_per_second:.2f}")
        with open(log_file, 'w') as file:
            file.write(f"{step} train {loss_accumulated.item():.6f}\n")

if ddp_is_enabled:
    distributed.destroy_process_group()