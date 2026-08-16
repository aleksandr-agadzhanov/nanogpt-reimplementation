import torch


def slice_input_output_batch(
    tokens: torch.Tensor, start_index: int, batch_size: int, context_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Slices batch_size * context_size + 1 tokens starting at start_index and splits
    them into (inputs, outputs), where outputs is inputs shifted right by one token.

    Args:
        tokens: A 1D tensor of token ids to slice from.
        start_index: Index into `tokens` where the batch starts.
        batch_size: Number of sequences in the batch.
        context_size: Number of tokens per sequence.

    Returns:
        inputs: Token ids of shape (batch_size, context_size).
        outputs: `inputs` shifted right by one token, i.e. outputs[:, s] is the
            next-token target for inputs[:, s], also of shape (batch_size, context_size).
    """
    tokens_per_batch = batch_size * context_size
    # Read one extra token beyond tokens_per_batch, since outputs need a target for
    # the last input position too.
    tokens_batch = tokens[start_index : start_index + tokens_per_batch + 1]
    inputs = tokens_batch[:-1].view(batch_size, context_size)
    outputs = tokens_batch[1:].view(batch_size, context_size)
    return inputs, outputs
