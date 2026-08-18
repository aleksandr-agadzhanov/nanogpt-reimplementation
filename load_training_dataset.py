import multiprocessing as mp
import os

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

SHARD_SIZE = 100_000_000  # 100M tokens per shard, ~100 shards for the 10BT sample
tokenizer = tiktoken.get_encoding("gpt2")
END_OF_TEXT_TOKEN = tokenizer._special_tokens["<|endoftext|>"]


def tokenize_dataset_to_shards(dataset, num_workers):
    """Tokenizes `dataset` across `num_workers` processes and writes the tokens out as fixed-size shards."""
    shard_index = 0
    shard_buffer = np.empty((SHARD_SIZE), dtype=np.uint16)
    tokens_in_shard = 0
    progress_bar = None

    with mp.Pool(num_workers) as pool:
        for document_tokens in pool.imap(tokenize_document, dataset, chunksize=16):
            # does the current shard have room for all of this document's tokens?
            if tokens_in_shard + len(document_tokens) < SHARD_SIZE:
                shard_buffer[
                    tokens_in_shard : tokens_in_shard + len(document_tokens)
                ] = document_tokens
                tokens_in_shard += len(document_tokens)

                if progress_bar is None:
                    progress_bar = tqdm(
                        total=SHARD_SIZE, unit="tokens", desc=f"Shard {shard_index}"
                    )
                progress_bar.update(len(document_tokens))

            else:
                # fill the current shard with as much of the document as fits, then write it out
                space_left_in_shard = SHARD_SIZE - tokens_in_shard
                shard_buffer[tokens_in_shard:] = document_tokens[:space_left_in_shard]
                progress_bar.update(space_left_in_shard)
                write_shard(shard_buffer, shard_index)
                shard_index += 1
                progress_bar = None

                # seed a fresh shard with whatever tokens didn't fit in the previous one
                remaining_tokens = document_tokens[space_left_in_shard:]
                shard_buffer[: len(remaining_tokens)] = remaining_tokens
                tokens_in_shard = len(remaining_tokens)

        # the dataset rarely ends exactly on a shard boundary, so flush the partial final shard
        if tokens_in_shard != 0:
            write_shard(shard_buffer[:tokens_in_shard], shard_index)


def tokenize_document(document):
    """Tokenizes a single document into a uint16 numpy array, prefixed with the end-of-text token."""
    tokens = [
        END_OF_TEXT_TOKEN
    ]  # delimits documents from one another in the token stream
    tokens.extend(tokenizer.encode_ordinary(document["text"]))
    tokens_np = np.array(tokens)
    # GPT-2's vocabulary fits in uint16 (up to 2^16 - 1 = 65535), so we can halve storage size
    return tokens_np.astype(np.uint16)


def write_shard(tokens, shard_index):
    """Writes a shard to disk; shard 0 is reserved for validation, every other shard is training data."""
    split = "val" if shard_index == 0 else "train"
    np.save(f"training_data/fineweb_edu_10bt_{split}_{shard_index:06d}", tokens)


def main():
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
