import os

import numpy as np
import torch


# TODO: add functionality to permute the data randomly in every single shard on every single new epoch, and to maybe even permute the shards
class ShardDataLoader:
    def __init__(
        self, folder_name, batch_size, context_size, process_rank, num_processes, split
    ):
        self.batch_size = batch_size
        self.context_size = context_size
        self.process_rank = process_rank
        self.num_processes = num_processes

        file_names = [
            file_name for file_name in os.listdir(folder_name) if split in file_name
        ]
        file_names = sorted(file_names)
        self.shard_paths = [
            os.path.join(folder_name, file_name) for file_name in file_names
        ]
        if process_rank == 0:
            print(f"Found {len(file_names)} shards for split - {split}")

        self.reset()

    def reset(self):
        self.current_shard = 0
        self.current_position = self.batch_size * self.context_size * self.process_rank
        self.tokens = ShardDataLoader.load_tokens(self.shard_paths[self.current_shard])

    def get_next_batch(self):
        # Without replacement
        batch_end_index = (
            self.current_position + self.batch_size * self.context_size + 1
        )
        tokens_batch = self.tokens[self.current_position : batch_end_index]
        inputs = tokens_batch[:-1].view(self.batch_size, self.context_size)
        outputs = tokens_batch[1:].view(self.batch_size, self.context_size)

        self.current_position = (
            self.current_position
            + self.batch_size * self.context_size * self.num_processes
        )

        if (
            self.current_position
            + self.batch_size * self.context_size * self.num_processes
            + 1
            > len(self.tokens)
        ):
            # Looping is introduced so that when we are out of shards, we can start from the first shard again
            self.current_shard = (self.current_shard + 1) % len(self.shard_paths)
            self.tokens = ShardDataLoader.load_tokens(self.shard_paths[self.current_shard])
            self.current_position = (
                self.batch_size * self.context_size * self.process_rank
            )

        return inputs, outputs

    @staticmethod
    def load_tokens(file_path):
        numpy_tokens = np.load(file_path)
        numpy_tokens = numpy_tokens.astype(np.int32)
        tokens = torch.tensor(numpy_tokens, dtype=torch.long)
        return tokens
