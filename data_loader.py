import tiktoken
import torch

class DataLoader:
    def __init__(self, file_name, batch_size, context_size):
        self.batch_size = batch_size
        self.context_size = context_size
        self.current_position = 0

        with open(f"training_datasets/{file_name}", 'r') as file:
            text = file.read()

        tokenizer = tiktoken.get_encoding("gpt2")
        tokens = tokenizer.encode(text)
        self.tokens = torch.tensor(tokens)

        num_batches = len(self.tokens) // (batch_size * context_size)
        print(f"Loaded {len(self.tokens)} tokens")
        print(f"This is {len(self.tokens)} // ({batch_size} * {context_size}) = {num_batches} batches")

    def get_next_batch(self):
        batch_end_index = self.current_position + self.batch_size * self.context_size + 1
        tokens_batch = self.tokens[self.current_position:batch_end_index]
        inputs = tokens_batch[:-1].view(self.batch_size, self.context_size)
        outputs = tokens_batch[1:].view(self.batch_size, self.context_size)

        self.current_position = batch_end_index - 1
        if self.current_position + self.batch_size * self.context_size + 1 > len(self.tokens):
            self.current_position = 0

        return inputs, outputs
