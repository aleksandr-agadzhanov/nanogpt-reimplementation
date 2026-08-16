from pathlib import Path

import tiktoken
import torch

from data_loaders.data_loader_utils import slice_input_output_batch

TRAINING_DATASETS_DIR = Path(__file__).resolve().parent.parent / "training_datasets"


class BasicDataLoader:
    """Tokenizes a text file once, then serves it back as fixed-size (input, output)
    batches for next-token-prediction training, looping back to the start once the
    tokens are exhausted."""

    def __init__(self, file_name: str, batch_size: int, context_size: int) -> None:
        """Reads and tokenizes a file from the training_datasets folder and
        initializes the batch cursor.

        Args:
            file_name: Name of a plain-text file inside the training_datasets folder.
            batch_size: Number of (input, output) sequences per batch.
            context_size: Number of tokens per sequence.
        """
        self.batch_size = batch_size
        self.context_size = context_size

        # Resolve file_name inside TRAINING_DATASETS_DIR and reject anything (e.g. an
        # absolute path or a "../" sequence) that would resolve outside of it.
        file_path = (TRAINING_DATASETS_DIR / file_name).resolve()
        if not file_path.is_relative_to(TRAINING_DATASETS_DIR):
            raise ValueError(
                f"file_name ({file_name!r}) must resolve to a path inside {TRAINING_DATASETS_DIR}"
            )

        # Read the provided file
        with open(file_path, "r") as file:
            text = file.read()

        # Tokenize the text
        tokenizer = tiktoken.get_encoding("gpt2")
        self.tokens = tokenizer.encode(text)
        self.tokens = torch.tensor(self.tokens, dtype=torch.long)

        # Every batch needs batch_size * context_size + 1 tokens (the +1 is the extra
        # target token for the last input position); fail fast with a clear message
        # rather than letting a later .view() call fail with a confusing shape error.
        self.tokens_per_batch = batch_size * context_size
        if len(self.tokens) < self.tokens_per_batch + 1:
            raise ValueError(
                f"file only contains {len(self.tokens)} tokens, but batch_size "
                f"({batch_size}) * context_size ({context_size}) + 1 = "
                f"{self.tokens_per_batch + 1} tokens are needed for a single batch"
            )

        # Initialize the index of the current token
        self.current_index = 0

    def get_next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the next (inputs, outputs) batch, looping back to the start of
        the tokens if there aren't enough tokens left for another full batch.

        Returns:
            inputs: Token ids of shape (batch_size, context_size).
            outputs: `inputs` shifted right by one token, i.e. outputs[:, s] is
                the next-token target for inputs[:, s], also of shape
                (batch_size, context_size).
        """
        # Call the utility function to slice the tokens into (inputs, outputs) batch
        inputs, outputs = slice_input_output_batch(
            self.tokens, self.current_index, self.batch_size, self.context_size
        )

        # Advance the current index by tokens_per_batch
        self.current_index = self.current_index + self.tokens_per_batch

        # If the next batch wouldn't have enough tokens left to read, loop back to
        # the start rather than reading (and silently truncating) past the end.
        if self.current_index + self.tokens_per_batch + 1 > len(self.tokens):
            self.current_index = 0

        return inputs, outputs
