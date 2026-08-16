import os
import sys
from pathlib import Path

import pytest
import tiktoken
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gpt import GPT

# Downloads a real GPT-2 checkpoint from HuggingFace, so this is opt-in rather than
# run by default; set RUN_NETWORK_TESTS=1 to enable it.
requires_network = pytest.mark.skipif(
    not os.environ.get("RUN_NETWORK_TESTS"),
    reason="Downloads real GPT-2 weights from HuggingFace; set RUN_NETWORK_TESTS=1 to run.",
)


@requires_network
def test_pretrained_gpt2_generates_plausible_continuations():
    """Loads real GPT-2 weights via `from_pretrained` and verifies that top-k
    sampled continuations are valid tokens and actually extend the prompt."""
    # Set the parameters for the test
    num_return_sequences = 5
    max_sequence_length = 30
    top_k = 50
    prompt = "Hello, I'm a language model,"

    # Determine the device to run the model on (CPU, CUDA, or MPS)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.mps.is_available():
        device = "mps"

    # Create an instance of the GPT model and load pretrained weights into it
    model = GPT.from_pretrained("gpt2")
    model.eval()
    model.to(device)

    # Encode the prompt into token IDs using the GPT-2 tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    tokens = tokenizer.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long)

    # Repeat the prompt across the batch dimension to sample several continuations at once.
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    tokens = tokens.to(device)

    # Fixed seed makes the sampled continuations reproducible
    torch.manual_seed(42)

    # Sample continuations until the desired length is reached
    while tokens.size(1) < max_sequence_length:
        with torch.no_grad():
            logits, _ = model(tokens, last_position_only=True)  # B x S x E -> B x 1 x V
            logits = logits.squeeze(1)  # B x 1 x V -> B x V, drop the size-1 sequence dim
            probabilities = logits.softmax(-1)

            # Sample from the top-k most likely tokens
            top_k_probabilities, top_k_indices = torch.topk(probabilities, top_k, -1)

            # Sample an index from the top-k probabilities for each sequence in the batch
            sampled_index = torch.multinomial(
                top_k_probabilities, 1
            )

            # Map the sampled index back to a real token ID using the top-k indices
            next_tokens = torch.gather(
                top_k_indices, -1, sampled_index
            )

            # Append the sampled token IDs to the existing sequences
            tokens = torch.cat((tokens, next_tokens), dim=-1)

    assert tokens.shape == (num_return_sequences, max_sequence_length)
    assert torch.all(
        (tokens >= 0) & (tokens < tokenizer.n_vocab)
    )  # every generated id is a valid token

    # Decode the generated token sequences back into text and verify that they extend the prompt
    for i in range(num_return_sequences):
        decoded = tokenizer.decode(tokens[i].tolist())
        print(f"Decoded sequence {i + 1}: {decoded}\n")
        assert decoded.startswith(prompt)
        assert len(decoded) > len(prompt)
