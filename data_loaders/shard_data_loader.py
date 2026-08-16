import os

import numpy as np
import tiktoken
import torch

from data_loaders.data_loader_utils import slice_input_output_batch

# Documents are delimited by this token within a shard (see load_training_dataset.py)
END_OF_TEXT_TOKEN = tiktoken.get_encoding("gpt2")._special_tokens["<|endoftext|>"]


class ShardDataLoader:
    """Serves fixed-size (input, output) batches for next-token-prediction training
    from a folder of pre-tokenized, pre-sharded .npy files, splitting each batch of
    microbatches evenly across multiple data-parallel processes.

    Every shard is assumed to contain only whole documents (no document spans two
    shards). Each time a shard is loaded, its documents are shuffled; each time a full
    pass over all shards completes, the shards themselves are reshuffled for the next
    pass, so that later epochs don't repeat the exact same batches as the first. Both
    kinds of shuffling are seeded deterministically (see `seed`) so that every process
    computes the identical shuffle independently, without any inter-process
    communication, and therefore always reads a disjoint slice of the same data as
    every other process.
    """

    def __init__(
        self,
        folder_name: str,
        batch_size: int,
        context_size: int,
        process_index: int,
        num_processes: int,
        split: str,
        seed: int = 42,
    ) -> None:
        """Finds and sorts a folder's shard files matching `split`, and loads the
        first shard.

        Args:
            folder_name: Path to the folder containing shard .npy files.
            batch_size: Number of (input, output) sequences per process, per batch.
            context_size: Number of tokens per sequence.
            process_index: Index of this process among `num_processes` data-parallel
                processes (e.g. a DDP rank); determines this process's offset within
                each batch of microbatches.
            num_processes: Total number of data-parallel processes sharing each
                shard's tokens.
            split: Substring that a shard's file name must contain to be included
                (e.g. "train" or "val").
            seed: Base seed for shard/document shuffling. Must be the same on every
                data-parallel process so that they independently shuffle identically.
        """
        self.batch_size = batch_size
        self.context_size = context_size
        self.process_index = process_index
        self.num_processes = num_processes

        # Advanced after each completed pass over all shards (see get_next_batch), so
        # that shuffles vary across passes while staying identical across processes.
        self.seed = seed

        # Number of tokens per microbatch, i.e. per process, per batch
        self.tokens_per_microbatch = batch_size * context_size

        file_names = [
            file_name for file_name in os.listdir(folder_name) if split in file_name
        ]
        # Sort the shard files so that every process sees them in the same order
        file_names = sorted(file_names)
        self.shard_paths = [
            os.path.join(folder_name, file_name) for file_name in file_names
        ]
        if not self.shard_paths:
            raise ValueError(
                f"no shard files matching split={split!r} found in {folder_name!r}"
            )
        self.shard_order = list(range(len(self.shard_paths)))

        if process_index == 0:
            print(f"Found {len(file_names)} shards for split - {split}")

        self.reset()

    def reset(self) -> None:
        """Rewinds to the first shard (in its current order) and this process's
        starting offset within it."""
        self.current_shard_index = 0
        self.current_index = self.tokens_per_microbatch * self.process_index
        self.shard_tokens = self._load_shard(self.current_shard_index)

    def get_next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns this process's next (inputs, outputs) batch, advancing the cursor
        by one batch (i.e. by every process's combined share of tokens). Moves to the
        next shard - reshuffling shard order first if a full pass just completed -
        once there isn't room left for another full batch in the current shard.

        Returns:
            inputs: Token ids of shape (batch_size, context_size).
            outputs: `inputs` shifted right by one token, i.e. outputs[:, s] is the
                next-token target for inputs[:, s], also of shape
                (batch_size, context_size).
        """
        inputs, outputs = slice_input_output_batch(
            self.shard_tokens, self.current_index, self.batch_size, self.context_size
        )

        self.current_index = (
            self.current_index + self.tokens_per_microbatch * self.num_processes
        )

        # Strip out this process's own offset to get the start of the next batch, so
        # that every process agrees on when a shard runs out of room and all of them
        # switch shards together, rather than at slightly different times.
        next_batch_start_index = (
            self.current_index - self.process_index * self.tokens_per_microbatch
        )
        if (
            next_batch_start_index + self.tokens_per_microbatch * self.num_processes + 1
            > len(self.shard_tokens)
        ):
            # Calculate the next shard index, wrapping around to 0 if this was the last shard.
            next_shard_index = (self.current_shard_index + 1) % len(self.shard_order)
            if next_shard_index == 0:
                # A full pass over all shards just completed; advance the seed and
                # reshuffle their order for the next pass. Every process advances its
                # seed identically, so they reshuffle identically without
                # communicating with each other.
                self.seed += 1
                generator = torch.Generator().manual_seed(self.seed)
                self.shard_order = [
                    self.shard_order[i]
                    for i in torch.randperm(
                        len(self.shard_order), generator=generator
                    ).tolist()
                ]
            self.current_shard_index = next_shard_index
            self.shard_tokens = self._load_shard(self.current_shard_index)
            self.current_index = self.tokens_per_microbatch * self.process_index

        return inputs, outputs

    def _load_shard(self, shard_index: int) -> torch.Tensor:
        """Loads a shard's tokens, validates it holds enough tokens for at least one
        batch of microbatches across all processes (to fail fast rather than let a later
        slice_input_output_batch call silently truncate and error in .view()), and
        shuffles the order of its documents."""
        original_index = self.shard_order[shard_index]
        shard_path = self.shard_paths[original_index]
        tokens = self._load_tokens_from_shard(shard_path)

        # If the shard doesn't contain enough tokens for all processes to, raise an error.
        tokens_per_batch = self.tokens_per_microbatch * self.num_processes
        if len(tokens) < tokens_per_batch + 1:
            raise ValueError(
                f"shard {shard_path!r} only contains {len(tokens)} tokens, but "
                f"batch_size * context_size * num_processes + 1 = "
                f"{tokens_per_batch + 1} tokens are needed for a single batch of microbatches"
            )

        # Get seed for this shard's document shuffle, derived deterministically.
        shard_seed = self._derive_seed(original_index)

        # Create a generator with the derived seed.
        generator = torch.Generator().manual_seed(shard_seed)

        # Use the generator to shuffle the order of documents within the shard.
        tokens = self._permute_documents(tokens, generator)
        return tokens

    def _derive_seed(self, part: int) -> int:
        """Deterministically combines `self.seed` with `part` into a single seed, so
        that e.g. different shards within the same pass get distinct document
        shuffles."""
        # A prime multiplier is used to reduce the chance of collisions between seeds
        # derived from different (seed, part) pairs.
        return self.seed * 1_000_003 + part

    @staticmethod
    def _permute_documents(
        tokens: torch.Tensor, generator: torch.Generator
    ) -> torch.Tensor:
        """Splits tokens into per-document chunks at END_OF_TEXT_TOKEN boundaries
        (every shard is assumed to contain only whole documents) and concatenates
        them back together in a random order."""
        # True at every position holding the token that starts a new document
        is_document_start = tokens == END_OF_TEXT_TOKEN

        # Get the indices of the start of each document, and the end of each document.
        # Each document ends where the next one starts; the last ends at the shard's end.
        start_indices = is_document_start.nonzero(as_tuple=True)[0].tolist()
        end_indices = start_indices[1:] + [len(tokens)]

        # Split the shard's tokens into a list of per-document tensors
        documents = [
            tokens[start:end] for start, end in zip(start_indices, end_indices)
        ]

        # Shuffle the order of the documents using the provided generator
        shuffled_indices = torch.randperm(len(documents), generator=generator).tolist()

        # Concatenate the shuffled documents back into a single tensor.
        tokens = torch.cat([documents[i] for i in shuffled_indices])

        return tokens

    @staticmethod
    def _load_tokens_from_shard(shard_path: str) -> torch.Tensor:
        """Loads a shard's tokens from disk as a 1D tensor of token ids."""
        numpy_tokens = np.load(shard_path)
        # GPT-2's vocabulary fits in uint16, so shards are stored compactly on disk;
        # widen to int32 first since torch has no native unsigned 16-bit tensor dtype
        numpy_tokens = numpy_tokens.astype(np.int32)
        tokens = torch.tensor(numpy_tokens, dtype=torch.long)
        return tokens
