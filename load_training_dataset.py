import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
from datasets import Dataset, load_dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent / "minbpe_reimplementation"))
from minbpe_tokenizers import RegexTokenizer

# 10 billion tokens / max 100M tokens per shard ~ 100 shards. This is a token count,
# not a byte count; each shard's on-disk size is roughly double this (uint16 tokens).
MAX_TOKENS_PER_SHARD = 100_000_000

# Initialize the custom tokenizer with the trained vocabulary.
tokenizer = RegexTokenizer(vocabulary_file_name="fineweb_edu_100mb_16384.pkl")

# The end-of-text token is used to delimit documents in the tokenized dataset.
END_OF_TEXT_TOKEN = tokenizer.special_tokens["<|endoftext|>"]


def tokenize_dataset_to_shards(dataset: Dataset, num_workers: int) -> None:
    """Tokenizes every document in `dataset` in parallel and writes the resulting
    tokens to fixed-size shard files on disk.

    Documents are never split across two shards: if a document doesn't fit in the
    remaining space of the current shard, the current shard is flushed as-is and the
    document becomes the start of the next one. This keeps document boundaries intact
    for `ShardDataLoader`, which shuffles whole shards.

    Args:
        dataset: An iterable of documents (each a mapping with a "text" key), e.g. a
            HuggingFace `Dataset`.
        num_workers: Number of worker processes used to tokenize documents in
            parallel.
    """
    shard_index = 0
    tokens_in_shard = 0
    progress_bar = None

    # Initialize the shard buffer to hold the tokens to be written to the current shard.
    shard_buffer = np.empty((MAX_TOKENS_PER_SHARD), dtype=np.uint16)

    # Use a multiprocessing pool to tokenize documents in parallel.
    with mp.Pool(num_workers) as pool:
        # Use imap to tokenizer each document in the dataset in parallel.
        # Chunk size of 16 is used to reduce the inter-process communication overhead.
        for document_tokens in pool.imap(tokenize_document, dataset, chunksize=16):
            # If any single document is too large to fit in any shard, raise an error.
            if len(document_tokens) > MAX_TOKENS_PER_SHARD:
                raise ValueError(
                    f"A document tokenized to {len(document_tokens)} tokens, which "
                    f"exceeds MAX_TOKENS_PER_SHARD ({MAX_TOKENS_PER_SHARD})"
                )

            # If this document doesn't fit, flush the current shard as-is (never
            # splitting a document across shards) and start a fresh one for it
            if tokens_in_shard + len(document_tokens) > MAX_TOKENS_PER_SHARD:
                write_shard(shard_buffer[:tokens_in_shard], shard_index)

                # Increment the shard index and reset the token count and progress bar
                shard_index += 1
                tokens_in_shard = 0
                progress_bar = None

            # Append this document's tokens right after the previous document's.
            shard_buffer[tokens_in_shard : tokens_in_shard + len(document_tokens)] = (
                document_tokens
            )

            # Update the token count
            tokens_in_shard += len(document_tokens)

            # Update the progress bar, creating it if this is the first document
            # in the current shard.
            if progress_bar is None:
                progress_bar = tqdm(
                    total=MAX_TOKENS_PER_SHARD,
                    unit="tokens",
                    desc=f"Shard {shard_index}",
                )
            progress_bar.update(len(document_tokens))

        # Most likely there will be tokens left in the buffer after the last document,
        # so flush them to a final shard.
        if tokens_in_shard != 0:
            write_shard(shard_buffer[:tokens_in_shard], shard_index)


def tokenize_document(document: dict) -> np.ndarray:
    """Tokenizes a single document's text, prefixed with the end-of-text token.

    Args:
        document: A mapping with a "text" key holding the document's raw text.

    Returns:
        A 1D array of token ids, as uint16 (custom tokenizer's vocabulary fits in 2^16 - 1).
    """
    # Start each document with the end-of-text token following GPT-2 convention.
    tokens = [END_OF_TEXT_TOKEN]

    # Tokenize the document's text and append it to the end-of-text token.
    tokens.extend(tokenizer.encode_non_special(document["text"]))

    # Convert to a numpy array and downcast to uint16 for storage efficiency.
    numpy_tokens = np.array(tokens)

    # The custom vocabulary has 16,384 tokens, which fits in uint16 (max 65,535).
    numpy_tokens = numpy_tokens.astype(np.uint16)

    return numpy_tokens


def write_shard(tokens: np.ndarray, shard_index: int) -> None:
    """Writes a shard's tokens to disk.

    Args:
        tokens: The shard's tokens.
        shard_index: The shard's position among all shards; shard 0 is reserved for
            validation, every other shard is training data.
    """
    # Reserve shard 0 for validation, every other shard is training data.
    split = "val" if shard_index == 0 else "train"

    # Write the shard to disk.
    np.save(f"training_shards/fineweb_edu_10bt_{split}_{shard_index:04d}", tokens)


def main() -> None:
    """Loads the fineweb-edu dataset and tokenizes it to shard files on disk."""
    # Load the HuggingFace fineweb-edu dataset
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train"
    )

    # Use half of the CPU cores for tokenization, leaving the other half free for other tasks.
    num_workers = max(1, os.cpu_count() // 2)

    # Tokenize the dataset and write it to shards.
    tokenize_dataset_to_shards(dataset, num_workers)


if __name__ == "__main__":
    main()
