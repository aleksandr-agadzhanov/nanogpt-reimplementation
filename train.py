import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import distributed, nn
from torch.nn.parallel import DistributedDataParallel as DDP

from data_loaders.shard_data_loader import ShardDataLoader
from gpt import GPT, GPTConfig
from hellaswag import get_most_likely_row, iterate_examples, render_example

sys.path.insert(0, str(Path(__file__).resolve().parent / "minbpe_reimplementation"))
from minbpe_tokenizers import RegexTokenizer

# torchrun --standalone --nproc_per_node=8 train.py

# Manually set the seed for Pytorch to ensure reproducibility across runs.
TORCH_MANUAL_SEED = 42

# Data loader configuration
TRAINING_DATA_FOLDER = "training_shards"
MICRO_BATCH_SIZE = 16  # for GPT-2 use 64, for GPT-3 use 32 # number of tokens processed by each process during a single batch
CONTEXT_SIZE = 64  # GPT-2 uses 1024, GPT-3 uses 2048 # maximum sequence length (number of positions) the model supports

# This is used to simulate a larger batch size than what can fit in GPU memory.
# To do that, gradients are accumulated over multiple micro-batches before
# performing an optimizer step.
# The total batch size must be set to MICRO_BATCH_SIZE * CONTEXT_SIZE * ddp_world_size.
BATCH_SIZE = 1024  # 524288

# Optimizer configuration.
WEIGHT_DECAY = 0.1
MAX_LEARNING_RATE = 6e-4  # could 3x this
MIN_LEARNING_RATE = 0.1 * MAX_LEARNING_RATE
NUM_WARMUP_STEPS = 10  # 715 = 375e6 / 2**19

# The gradient norm is clipped to this value to avoid exploding gradients during training.
GRADIENT_CLIP_NORM = 1.0

# Whether to use torch.compile() to compile the model for faster training.
# TODO Check for interference with eval and generation.
USE_COMPILE = False  # compilation interferes with eval and generation

# Step related configuration.
NUM_TRAIN_STEPS = 50  # 19073 = 2**19 / 10e9 # number of training steps to run
EVAL_INTERVAL_STEPS = 250  # how many steps to run between evaluations of the model
VAL_STEPS = 20  # how many steps to run during validation
CHECKPOINT_INTERVAL_STEPS = (
    5000  # how many steps to run between saving model checkpoints
)

# Logging configuration.
LOG_DIRECTORY = "logs"  # directory to save logs and model checkpoints
LOG_FILE_NAME = (
    "train_gpt_log.txt"  # name of the log file to save training and evaluation metrics
)

# Checkpointing configuration.
CHECKPOINT_DIRECTORY = "checkpoints"  # directory to save model checkpoints

# Generation configuration.
GENERATION_PROMPT = "Hello, I'm a language model,"  # prompt to use
GENERATION_NUM_RETURN_SEQUENCES = 4  # number of sequences to generate for each prompt
GENERATION_MAX_LENGTH = 32  # maximum length of the generated sequences (in tokens)
GENERATION_TOP_K = 50  # number of top-k tokens to sample from during generation


def setup_ddp() -> tuple[bool, int, int, int, bool, str]:
    """Detects and initializes DistributedDataParallel (DDP) if launched via torchrun.

    Reads the "RANK", "LOCAL_RANK", and "WORLD_SIZE" environment variables that
    torchrun sets on every process to determine whether this is a distributed run.
    If so, initializes the NCCL process group and pins this process to its assigned
    GPU. Otherwise, falls back to a single process on the best available device
    (CUDA, then MPS, then CPU).

    Returns:
        A tuple of:
            is_distributed: Whether this process is part of a torchrun DDP launch.
            ddp_rank: This process's global rank across all DDP processes.
            ddp_local_rank: This process's rank among the DDP processes on this node;
                also selects which local GPU this process uses.
            ddp_world_size: Total number of DDP processes.
            is_master_process: Whether this process should handle logging/checkpointing
                (rank 0 in DDP, or the only process otherwise).
            device: The device string this process should run on (e.g. "cuda:0", "mps", "cpu").

    Raises:
        RuntimeError: If launched via torchrun but no CUDA device is available,
            since the NCCL backend requires CUDA.
    """
    # Detect whether this is a distributed run.
    is_distributed = "RANK" in os.environ

    if is_distributed:
        # NCCL backend requires CUDA, so raise an error if no CUDA device is available.
        if not torch.cuda.is_available():
            raise RuntimeError("DDP requires CUDA, but no CUDA device is available.")

        # Initialize the NCCL process group.
        distributed.init_process_group(backend="nccl")

        # Extract the DDP properties.
        ddp_rank = int(
            os.environ["RANK"]
        )  # Global rank of this process across all DDP processes.
        ddp_local_rank = int(
            os.environ["LOCAL_RANK"]
        )  # Rank of this process among the DDP processes on this node
        ddp_world_size = int(
            os.environ["WORLD_SIZE"]
        )  # Total number of DDP processes across all nodes.

        # Pin this process to its assigned GPU.
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)

        # Determine if this process is the master process.
        is_master_process = ddp_rank == 0
    else:
        # Set the default values for a non-distributed run.
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        is_master_process = True

        # Pick the best available device: CUDA, then MPS, then CPU.
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.mps.is_available():
            device = "mps"

    if is_master_process:
        print(f"Using device - {device}")
        print(
            f"Distributed: {is_distributed}, "
            f"DDP rank: {ddp_rank}, "
            f"DDP local rank: {ddp_local_rank}, "
            f"DDP world size: {ddp_world_size}"
        )

    return (
        is_distributed,
        ddp_rank,
        ddp_local_rank,
        ddp_world_size,
        is_master_process,
        device,
    )


def set_random_seed(seed: int) -> None:
    """Set the random seed for PyTorch and all available accelerator backends.

    Args:
        seed: The value used to initialize the random number generators.
    """
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if torch.mps.is_available():
        torch.mps.manual_seed(seed)


def get_num_gradient_accumulation_steps(
    batch_size: int,
    micro_batch_size: int,
    context_size: int,
    ddp_world_size: int,
    is_master_process: bool,
) -> int:
    """Calculate and optionally report gradient accumulation steps.

    Args:
        batch_size: Target global batch size in tokens.
        micro_batch_size: Number of sequences in each micro-batch per process.
        context_size: Number of tokens in each sequence.
        ddp_world_size: Number of processes participating in training.
        is_master_process: Whether this process should print configuration details.

    Returns:
        The number of micro-batches to accumulate before each optimizer step.

    Raises:
        ValueError: If the global batch size is not divisible by the tokens in a
            single distributed micro-batch.
    """
    # Calculate the number of tokens in a single micro-batch across all DDP processes.
    tokens_per_micro_batch = micro_batch_size * context_size * ddp_world_size

    # Validate that the global batch size is divisible by the tokens in a single
    # distributed micro-batch. If not, raise an error.
    if batch_size % tokens_per_micro_batch != 0:
        raise ValueError(
            "batch_size must be divisible by "
            "micro_batch_size * context_size * ddp_world_size: "
            f"{batch_size} is not divisible by {tokens_per_micro_batch}."
        )

    # Calculate the number of micro-batches to accumulate before each optimizer step.
    num_gradient_accumulation_steps = batch_size // tokens_per_micro_batch

    if is_master_process:
        print(
            "Gradient accumulation: "
            f"batch_size={batch_size}, "
            f"micro_batch_size={micro_batch_size}, "
            f"context_size={context_size}, "
            f"ddp_world_size={ddp_world_size}, "
            f"steps={num_gradient_accumulation_steps}"
        )

    return num_gradient_accumulation_steps


def get_learning_rate(step: int) -> float:
    """Return the learning rate for a training step.

    The schedule linearly warms up from zero to the maximum learning rate, then
    follows cosine decay toward the minimum learning rate.

    Args:
        step: The zero-based training step.

    Returns:
        The learning rate to use for the given step.
    """
    # Linearly increase the learning rate during the warmup period.
    if step < NUM_WARMUP_STEPS:
        return MAX_LEARNING_RATE * (step + 1) / NUM_WARMUP_STEPS

    # Keep the learning rate at its minimum after training has finished.
    if step >= NUM_TRAIN_STEPS:
        return MIN_LEARNING_RATE

    # Cosine-decay the learning rate between the warmup and final steps.
    # The decay ratio starts at 0 after warmup and reaches 1 at the final training step.
    decay_ratio = (step - NUM_WARMUP_STEPS) / (NUM_TRAIN_STEPS - 1 - NUM_WARMUP_STEPS)

    # The cosine coefficient starts at 1 when decay_ratio is 0 after warmup and reaches
    # 0 when decay_ratio is 1 at the final training step.
    coefficient = 0.5 * (1 + math.cos(math.pi * decay_ratio))

    # The learning rate starts at the maximum after warmup and decays to the minimum at
    # the final training step.
    return MIN_LEARNING_RATE + coefficient * (MAX_LEARNING_RATE - MIN_LEARNING_RATE)


def initialize_log_file(log_directory: str, log_file_name: str) -> str:
    """Create the log directory and initialize an empty log file.

    Args:
        log_directory: Directory in which to store the log file.
        log_file_name: Name of the log file to create.

    Returns:
        The path to the initialized log file.
    """
    os.makedirs(log_directory, exist_ok=True)
    log_file = os.path.join(log_directory, log_file_name)
    with open(log_file, "w"):
        pass
    return log_file


def write_log(log_file: str, message: str) -> None:
    """Append a message followed by a newline to a log file.

    Args:
        log_file: Path to the log file.
        message: Text to append to the log file.
    """
    with open(log_file, "a") as file:
        file.write(f"{message}\n")


def save_checkpoint(
    model: nn.Module,
    config: GPTConfig,
    checkpoint_directory: str,
    step: int,
    val_loss: torch.Tensor,
) -> str:
    """Save model parameters and metadata to a checkpoint file.

    Args:
        model: The unwrapped model whose parameters should be saved.
        config: Configuration used to construct the model.
        checkpoint_directory: Directory in which to save the checkpoint.
        step: Training step associated with the checkpoint.
        val_loss: Validation loss associated with the checkpoint.

    Returns:
        The path to the saved checkpoint file.
    """
    os.makedirs(checkpoint_directory, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_directory, f"gpt_{step:05d}.pt")
    checkpoint = {
        "model": model.state_dict(),
        "config": config,
        "step": step,
        "val_loss": val_loss.item(),
    }
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def run_validation(
    model: nn.Module,
    val_loader: ShardDataLoader,
    val_steps: int,
    device: str,
    is_distributed: bool,
) -> torch.Tensor:
    """Evaluate the model and return the mean validation loss.

    Args:
        model: The model to evaluate.
        val_loader: Data loader that provides validation batches.
        val_steps: Number of validation batches to evaluate.
        device: Device on which the model and validation batches reside.
        is_distributed: Whether to average the loss across DDP processes.

    Returns:
        The mean validation loss across the requested batches and processes.
    """
    # Record whether the model was in training mode before evaluation
    # so that it can be restored afterward.
    was_training = model.training

    # Set the model to evaluation mode to disable dropout and other training-specific behavior.
    model.eval()

    # Reset the validation data loader and shuffle the documents inside validation shards.
    val_loader.reset()

    # Initialize the accumulated validation loss to zero on the specified device.
    val_loss_accumulated = torch.zeros((), device=device)

    # Extract the device type (e.g., "cuda", "mps", "cpu") from the device string for autocast.
    device_type = torch.device(device).type

    # Use a try-finally block to ensure that the model's training state is restored.
    try:
        # Evaluate the model without computing gradients to save memory and computation.
        with torch.no_grad():
            for _ in range(val_steps):
                # Get the next batch of validation data and move it to the specified device.
                inputs, outputs = val_loader.get_next_batch()
                inputs = inputs.to(device)
                outputs = outputs.to(device)
                # Use autocast to reduce memory usage and speed up evaluation
                # by using lower precision for matrix multiplications.
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    # Compute the model's predictions and loss for the current batch.
                    _, loss = model(inputs, outputs)
                # Normalize the loss by the number of validation steps
                loss = loss / val_steps
                # Accumulate the normalized loss for averaging across all processes/
                val_loss_accumulated = val_loss_accumulated + loss.detach()

        # Average the per-process validation means across all DDP processes.
        if is_distributed:
            distributed.all_reduce(val_loss_accumulated, op=distributed.ReduceOp.AVG)
    finally:
        model.train(was_training)

    return val_loss_accumulated


def main() -> None:
    # Setup distributed data parallel (DDP) if available, otherwise fall back to a single process.
    (
        is_distributed,
        ddp_rank,
        ddp_local_rank,
        ddp_world_size,
        is_master_process,
        device,
    ) = setup_ddp()

    # Calculate the number of steps over which gradients will be accumulated before each
    # optimizer step to simulate a larger global batch size. This is equivalent to the
    # number of micro-batches processed by all DDP processes.
    num_gradient_accumulation_steps = get_num_gradient_accumulation_steps(
        BATCH_SIZE,
        MICRO_BATCH_SIZE,
        CONTEXT_SIZE,
        ddp_world_size,
        is_master_process,
    )

    # Set the random seed for reproducibility across runs.
    set_random_seed(TORCH_MANUAL_SEED)

    # Optimize the training by reducing the precision of matrix multiplications to "high"
    # from the default "highest".
    torch.set_float32_matmul_precision("high")

    # Initialize the training and validation data loaders which will serve batches to the model.
    train_loader = ShardDataLoader(
        TRAINING_DATA_FOLDER,
        MICRO_BATCH_SIZE,
        CONTEXT_SIZE,
        ddp_rank,
        ddp_world_size,
        "train",
    )

    val_loader = ShardDataLoader(
        TRAINING_DATA_FOLDER,
        MICRO_BATCH_SIZE,
        CONTEXT_SIZE,
        ddp_rank,
        ddp_world_size,
        "val",
    )

    # Initialize the custom tokenizer.
    tokenizer = RegexTokenizer(vocabulary_file_name="fineweb_edu_100mb_16384.pkl")

    # Create the GPT model and move it to the appropriate device.
    gpt_config = GPTConfig()
    model = GPT(gpt_config)
    model.to(device)

    # Optionally compile the model for faster training.
    if USE_COMPILE:
        model = torch.compile(model)

    # Wrap the model in DistributedDataParallel (DDP) if running in a distributed setting.
    if is_distributed:
        model = DDP(model, device_ids=[ddp_local_rank])

    # Get the raw model (unwrapped) for optimizer configuration and other operations.
    raw_model = model.module if is_distributed else model

    # Create the optimizer with weight decay and learning rate settings.
    optimizer = raw_model.get_configured_optimizer(
        weight_decay=WEIGHT_DECAY, learning_rate=MAX_LEARNING_RATE, device=device
    )

    # Initialize the log file to record training and evaluation metrics.
    log_file = initialize_log_file(LOG_DIRECTORY, LOG_FILE_NAME)

    # Main training loop
    for step in range(NUM_TRAIN_STEPS):
        # Record the start time of the step to measure training speed.
        t0 = time.time()

        # Create a flag to indicate whether this is the last training step
        # used for evaluation and checkpointing.
        is_last_step = step == NUM_TRAIN_STEPS - 1

        # Every EVAL_INTERVAL_STEPS or on the last step, run validation.
        if step % EVAL_INTERVAL_STEPS == 0 or is_last_step:
            val_loss_accumulated = run_validation(
                model, val_loader, VAL_STEPS, device, is_distributed
            )
            # If this is the master process, print and log the validation loss.
            if is_master_process:
                validation_message = (
                    f"Step {step:05d} | validation loss: "
                    f"{val_loss_accumulated.item():.4f}"
                )
                print(validation_message)
                write_log(log_file, validation_message)

        # If this is the master process, every CHECKPOINT_INTERVAL_STEPS (apart from the
        # first step) or on the last step, save a model checkpoint.
        if is_master_process and (
            step > 0 and (step % CHECKPOINT_INTERVAL_STEPS == 0 or is_last_step)
        ):
            checkpoint_path = save_checkpoint(
                raw_model,
                raw_model.config,
                CHECKPOINT_DIRECTORY,
                step,
                val_loss_accumulated,
            )
            checkpoint_message = (
                f"Step {step:05d} | checkpoint saved: {checkpoint_path}"
            )
            print(checkpoint_message)
            write_log(log_file, checkpoint_message)

        # Hellaswag evaluation
        if (step % EVAL_INTERVAL_STEPS == 0 or is_last_step) and not USE_COMPILE:
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
                if is_distributed:
                    num_total = torch.tensor(num_total, dtype=torch.long, device=device)
                    num_correct_norm = torch.tensor(
                        num_correct_norm, dtype=torch.long, device=device
                    )
                    distributed.all_reduce(num_total, op=distributed.ReduceOp.SUM)
                    distributed.all_reduce(
                        num_correct_norm, op=distributed.ReduceOp.SUM
                    )
                    num_total = num_total.item()
                    num_correct_norm = num_correct_norm.item()
                acc_norm = num_correct_norm / num_total
                if is_master_process:
                    hellaswag_message = (
                        f"Step {step:05d} | HellaSwag accuracy: "
                        f"{acc_norm:.4f} ({num_correct_norm}/{num_total})"
                    )
                    print(hellaswag_message)
                    write_log(log_file, hellaswag_message)

        # Generation - REQUIRES DISABLING MODEL COMPLILE, SO DON'T DO IT TO TRAIN (test first, maybe not)
        if (
            (step > 0 and step % EVAL_INTERVAL_STEPS == 0) or is_last_step
        ) and not USE_COMPILE:
            model.eval()
            tokens = tokenizer.encode(GENERATION_PROMPT)
            tokens = torch.tensor(tokens, dtype=torch.long)
            tokens = tokens.unsqueeze(0).repeat(GENERATION_NUM_RETURN_SEQUENCES, 1)
            xgen = tokens.to(device)
            sample_rng = torch.Generator(device=device)
            sample_rng.manual_seed(TORCH_MANUAL_SEED + ddp_rank)
            while xgen.size(1) < GENERATION_MAX_LENGTH:
                with torch.no_grad():
                    with torch.autocast(device_type=device, dtype=torch.bfloat16):
                        logits, loss = model(xgen, last_position_only=True)
                    logits = logits[:, -1, :]
                    probs = logits.softmax(-1)
                    topk_probs, topk_indices = torch.topk(probs, GENERATION_TOP_K, -1)
                    ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
                    xcol = torch.gather(topk_indices, -1, ix)
                    x = torch.cat((xgen, xcol), dim=-1)
            for i in range(GENERATION_NUM_RETURN_SEQUENCES):
                tokens = x[i, :GENERATION_MAX_LENGTH].tolist()
                decoded = tokenizer.decode(tokens)
                print(f"Rank {ddp_rank}, sample {i + 1}: {decoded}")

        optimizer.zero_grad()

        loss_accumulated = 0.0
        for micro_step in range(num_gradient_accumulation_steps):
            x, y = train_loader.get_next_batch()
            x = x.to(device)
            y = y.to(device)
            is_last_micro_step = micro_step == num_gradient_accumulation_steps - 1

            if is_distributed:
                model.require_backward_grad_sync = is_last_micro_step

            # Only sync gradients across processes on the last micro-step
            sync_context = (
                model.no_sync()
                if is_distributed and not is_last_micro_step
                else nullcontext()
            )
            with sync_context:
                # Optimization 2
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                # Below is needed to account for gradient accumulation, so that we don't simply sum the gradients but take their mean
                loss = loss / num_gradient_accumulation_steps
                loss_accumulated = loss_accumulated + loss
                loss.backward()

        # To synchronize the loss across all processed
        if is_distributed:
            distributed.all_reduce(loss_accumulated, op=distributed.ReduceOp.AVG)

        # Clipping the gradient l2-norm to 1 so that if we get unlucky with a batch, the model shock coming from high gradients is avoided
        norm = nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)

        lr = get_learning_rate(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.step()
        # torch.cuda.synchronize() # wait for the GPU to finish work
        t1 = time.time()
        dt = (t1 - t0) * 1000  # ms
        tokens_processed = (
            train_loader.batch_size
            * train_loader.context_size
            * num_gradient_accumulation_steps
            * ddp_world_size
        )
        tokens_per_second = tokens_processed / (t1 - t0)

        if is_master_process:
            training_message = (
                f"Step {step:05d} | train loss: {loss_accumulated.item():.6f} | "
                f"lr: {lr:.4e} | grad norm: {norm:.4f} | "
                f"time: {dt:.2f} ms | tokens/s: {tokens_per_second:.2f}"
            )
            print(training_message)
            write_log(log_file, training_message)

    if is_distributed:
        distributed.destroy_process_group()


if __name__ == "__main__":
    main()
