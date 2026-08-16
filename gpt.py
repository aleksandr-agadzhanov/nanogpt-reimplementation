from __future__ import annotations

import inspect
import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from transformers import GPT2LMHeadModel


@dataclass
class GPTConfig:
    """Hyperparameters defining a GPT model's architecture.

    Note: attention_head_size * num_attention_heads must equal embedding_size.
    """

    # Size-related constants
    vocabulary_size: int = 16384  # Number of unique tokens in the vocabulary
    context_size: int = (
        1024  # Max sequence length (number of positions) the model supports
    )
    embedding_size: int = (
        768  # Dimensionality of token/position embeddings and residual stream
    )
    attention_head_size: int = (
        64  # Dimensionality of each attention head's query/key/value vectors
    )

    # Transformer-related constants
    num_attention_heads: int = (
        12  # Number of attention heads per multi-head attention layer
    )
    num_transformer_blocks: int = 12  # Number of stacked transformer blocks

    # Ratios
    feed_forward_ratio: int = (
        4  # Multiplier for the feed-forward layer's hidden dimension
    )

    def __post_init__(self):
        """Validates field values and cross-field constraints after dataclass initialization."""
        # All size/count fields must be positive.
        for field_name in (
            "vocabulary_size",
            "context_size",
            "embedding_size",
            "attention_head_size",
            "num_transformer_blocks",
            "num_attention_heads",
            "feed_forward_ratio",
        ):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive")

        # Concatenating all attention heads' outputs must reconstruct the full embedding.
        if self.embedding_size != self.attention_head_size * self.num_attention_heads:
            raise ValueError(
                "embedding_size must equal attention_head_size * num_attention_heads "
                f"({self.embedding_size} != {self.attention_head_size} * {self.num_attention_heads})"
            )

        # These fields must be powers of 2 for hardware/kernel efficiency.
        for field_name in (
            "context_size",
            "attention_head_size",
            "feed_forward_ratio",
        ):
            value = getattr(self, field_name)
            if not math.log2(value).is_integer():
                raise ValueError(f"{field_name} must be a power of 2 ({value} is not)")


class ResidualProjection(nn.Linear):
    """Linear layer at the end of a residual branch; scaled at init to control residual stream variance."""


class GPT(nn.Module):
    """A GPT-2-style decoder-only transformer language model.

    Embeds tokens and positions, runs them through a stack of causal
    TransformerBlocks, and projects the result back to vocabulary logits via
    a language model head whose weights are tied to the token embedding.
    """

    def __init__(self, config: GPTConfig):
        """Builds the embeddings, transformer stack, and output head defined by `config`.

        Args:
            config: A validated GPTConfig describing the model's architecture.
        """
        super().__init__()

        self.config = config

        # Initialize the model architecture
        self.token_embedding = nn.Embedding(
            config.vocabulary_size, config.embedding_size
        )
        self.position_embedding = nn.Embedding(
            config.context_size, config.embedding_size
        )
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_transformer_blocks)]
        )
        self.layer_norm = nn.LayerNorm(config.embedding_size)
        self.language_model_head = nn.Linear(
            config.embedding_size, config.vocabulary_size, bias=False
        )

        # Sharing weights between token embedding and language model head layers as per
        # GPT-2 conventions.
        self.token_embedding.weight = self.language_model_head.weight

        # Initialize the buffer of position indices which will be used for the position
        # embedding layer. It is not persistent: derived solely from context_size, so it must
        # not be saved to (and required to match on load from) a checkpoint's state_dict.
        self.register_buffer(
            "positions",
            torch.arange(config.context_size, dtype=torch.long),
            persistent=False,
        )

        # Re-initializes every submodule's weights per GPT-2 conventions, including the
        # tied token_embedding/language_model_head weight (initialized twice, harmlessly).
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initializes one submodule's parameters per GPT-2 conventions.

        Invoked on every submodule via `self.apply(self._init_weights)`. LayerNorm
        modules are left untouched since PyTorch already initializes them to
        weight=1, bias=0.
        """
        if isinstance(module, nn.Linear):
            # GPT-2's fixed std=0.02 replaces PyTorch's default Kaiming-uniform init,
            # so every Linear layer starts with the same small, controlled weight scale.
            std = 0.02
            if isinstance(module, ResidualProjection):
                # Each transformer block adds two residual contributions (attention output
                # and MLP output), so unscaled projections would make the residual stream's
                # variance grow with depth; dividing by sqrt(2 * num_transformer_blocks)
                # keeps it roughly constant regardless of model depth.
                std = std / math.sqrt(2 * self.config.num_transformer_blocks)
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                # Zero bias avoids injecting an arbitrary constant offset at init.
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Token/position embeddings use the same base std as non-residual Linear layers.
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        tokens: torch.Tensor,
        targets: torch.Tensor | None = None,
        last_position_only: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Runs a forward pass and optionally computes the next-token loss.

        Args:
            tokens: Token ids of shape (B, S); S must not exceed config.context_size.
            targets: Optional next-token ids of shape (B, S), aligned so that
                targets[:, s] is the label for tokens[:, s].
            last_position_only: If True, skips projecting every position to
                vocabulary logits and only does so for the last one; useful during
                generation, where only the next token's logits are ever needed.
                Must not be set when `targets` is given, since loss needs every
                position's logits.

        Returns:
            logits of shape (B, S, V) (or (B, 1, V) if `last_position_only`), and
            loss (a scalar tensor, or None if `targets` was not given).
        """
        if last_position_only and targets is not None:
            raise ValueError("last_position_only cannot be used together with targets")

        _, sequence_size = tokens.size()
        # If the sequence length exceeds the model's context size, it cannot be processed.
        if sequence_size > self.config.context_size:
            raise ValueError(
                f"sequence_size ({sequence_size}) exceeds context_size "
                f"({self.config.context_size})"
            )

        token_embeddings = self.token_embedding(tokens)  # B x S -> B x S x E

        # Move the position indices buffer to the same device as the input tokens.
        positions = self.positions.to(tokens.device)  # C
        positions = positions[
            :sequence_size
        ]  # C -> S (truncate to this batch's sequence length)
        position_embeddings = self.position_embedding(positions)  # S -> S x E

        # S x E broadcasts against B x S x E, adding the same position embeddings to every batch element.
        hidden_states = token_embeddings + position_embeddings  # B x S x E

        for transformer_block in self.transformer_blocks:
            hidden_states = transformer_block(hidden_states)  # B x S x E

        hidden_states = self.layer_norm(hidden_states)  # B x S x E

        if last_position_only:
            hidden_states = hidden_states[:, -1:, :]  # B x S x E -> B x 1 x E

        logits = self.language_model_head(hidden_states)  # B x S x E -> B x S x V or B x 1 x E -> B x 1 x V

        loss = None
        if targets is not None:
            # Flatten batch and sequence dims so cross_entropy scores each position independently:
            # (B x S x V -> (B*S) x V) against (B x S -> B*S).
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            )

        return logits, loss

    def get_configured_optimizer(
        self,
        weight_decay: float,
        learning_rate: float,
        device: str,
        is_master_process: bool,
    ) -> torch.optim.Optimizer:
        """Builds an AdamW optimizer with separate decay rules for weights and biases.

        Args:
            weight_decay: L2 regularization coefficient applied only to matrix-like parameters.
            learning_rate: Step size for AdamW.
            device: Device on which the model runs; used to decide whether fused AdamW is allowed.
            is_master_process: Whether this process is the master process; used to control printing.

        Returns:
            An AdamW optimizer configured with two parameter groups: one for decayed weights,
            and one for non-decayed biases / layer-norm parameters.
        """
        # Keep only trainable parameters; the optimizer should ignore frozen tensors.
        parameters = [
            parameter
            for _, parameter in self.named_parameters()
            if parameter.requires_grad
        ]

        # Weight matrices and embeddings are typically the only parameters that should receive
        # weight decay; biases and LayerNorm scale/bias values are usually left unregularized.
        parameters_to_decay = [
            parameter for parameter in parameters if parameter.dim() >= 2
        ]
        parameters_not_to_decay = [
            parameter for parameter in parameters if parameter.dim() < 2
        ]
        parameter_groups = [
            {"params": parameters_to_decay, "weight_decay": weight_decay},
            {"params": parameters_not_to_decay, "weight_decay": 0.0},
        ]

        # These prints are helpful for debugging training setup and verifying the expected split.
        if is_master_process:
            num_decayed_parameters = sum(
                parameter.numel() for parameter in parameters_to_decay
            )
            num_non_decayed_parameters = sum(
                parameter.numel() for parameter in parameters_not_to_decay
            )
            print(
                f"Decayed tensors - {len(parameters_to_decay)}, decayed parameters - {num_decayed_parameters:,}"
            )
            print(
                f"Non-decayed tensors - {len(parameters_not_to_decay)}, non-decayed parameters - {num_non_decayed_parameters:,}"
            )

        # Fused AdamW is only supported on CUDA and only when the PyTorch build exposes the flag.
        is_fused_available = "fused" in inspect.signature(
            torch.optim.AdamW
        ).parameters and str(device).startswith("cuda")
        if is_master_process:
            print(f"Using fused Adam - {is_fused_available}")

        # Betas and epsilon parameters are set according to the original GPT-2 implementation.
        optimizer = torch.optim.AdamW(
            parameter_groups,
            lr=learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            fused=is_fused_available,
        )
        return optimizer

    @classmethod
    def from_pretrained(cls, model_type: str) -> GPT:
        """Loads pretrained GPT-2 weights from HuggingFace into this implementation.

        Adapted from https://github.com/karpathy/build-nanogpt.git.

        Args:
            model_type: One of "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl".

        Returns:
            A GPT instance whose parameters are copied from the corresponding
            HuggingFace `GPT2LMHeadModel` checkpoint.
        """
        # Architecture dimensions for each released GPT-2 checkpoint size, keyed by
        # the HuggingFace model_type string.
        pretrained_model_configs = {
            "gpt2": {
                "num_transformer_blocks": 12,
                "num_attention_heads": 12,
                "embedding_size": 768,
            },  # 124M params
            "gpt2-medium": {
                "num_transformer_blocks": 24,
                "num_attention_heads": 16,
                "embedding_size": 1024,
            },  # 350M params
            "gpt2-large": {
                "num_transformer_blocks": 36,
                "num_attention_heads": 20,
                "embedding_size": 1280,
            },  # 774M params
            "gpt2-xl": {
                "num_transformer_blocks": 48,
                "num_attention_heads": 25,
                "embedding_size": 1600,
            },  # 1558M params
        }

        if model_type not in pretrained_model_configs:
            raise ValueError(
                f"Unknown model_type {model_type!r}; expected one of "
                f"{sorted(pretrained_model_configs)}"
            )

        # The HuggingFace GPT-2 checkpoints all use the same vocabulary size and context size.
        config_args = dict(pretrained_model_configs[model_type])
        config_args["vocabulary_size"] = 50257  # always 50257 for GPT model checkpoints
        config_args["context_size"] = 1024  # always 1024 for GPT model checkpoints
        embedding_size = config_args["embedding_size"]
        num_attention_heads = config_args["num_attention_heads"]

        # Validate that the embedding size is evenly divisible by the number of attention heads.
        if embedding_size % num_attention_heads != 0:
            raise ValueError(
                f"embedding_size ({embedding_size}) is not evenly divisible by "
                f"num_attention_heads ({num_attention_heads})"
            )
        config_args["attention_head_size"] = embedding_size // num_attention_heads

        # Build a GPT instance using one of the above pretrained model configs.
        config = GPTConfig(**config_args)
        model = cls(config)

        # Load the HuggingFace GPT-2 checkpoint.
        model_huggingface = GPT2LMHeadModel.from_pretrained(model_type)
        state_dict_huggingface = model_huggingface.state_dict()

        # Disables autograd tracking so copy_ doesn't try to record these in-place writes.
        with torch.no_grad():
            # Copy token embedding, position embedding, layer norm, and language model head weights and biases.
            model.token_embedding.weight.copy_(
                state_dict_huggingface["transformer.wte.weight"]
            )
            model.position_embedding.weight.copy_(
                state_dict_huggingface["transformer.wpe.weight"]
            )
            model.layer_norm.weight.copy_(
                state_dict_huggingface["transformer.ln_f.weight"]
            )
            model.layer_norm.bias.copy_(state_dict_huggingface["transformer.ln_f.bias"])
            model.language_model_head.weight.copy_(
                state_dict_huggingface["lm_head.weight"]
            )

            # Copy each transformer block's weights and biases.
            for transformer_block_index in range(config.num_transformer_blocks):
                cls._copy_pretrained_block_weights(
                    transformer_block=model.transformer_blocks[transformer_block_index],
                    state_dict_huggingface=state_dict_huggingface,
                    huggingface_prefix=f"transformer.h.{transformer_block_index}",
                )

        return model

    @staticmethod
    def _copy_pretrained_block_weights(
        transformer_block: TransformerBlock,
        state_dict_huggingface: dict,
        huggingface_prefix: str,
    ) -> None:
        """Copies one HuggingFace GPT-2 block's weights into `transformer_block`.

        Adapted from https://github.com/karpathy/build-nanogpt.git.
        """
        # Layer norms.
        transformer_block.layer_norm_1.weight.copy_(
            state_dict_huggingface[f"{huggingface_prefix}.ln_1.weight"]
        )
        transformer_block.layer_norm_1.bias.copy_(
            state_dict_huggingface[f"{huggingface_prefix}.ln_1.bias"]
        )
        transformer_block.layer_norm_2.weight.copy_(
            state_dict_huggingface[f"{huggingface_prefix}.ln_2.weight"]
        )
        transformer_block.layer_norm_2.bias.copy_(
            state_dict_huggingface[f"{huggingface_prefix}.ln_2.bias"]
        )

        # HuggingFace's Conv1D stores weights as (in_features, out_features), the transpose of
        # nn.Linear's (out_features, in_features), so .t() makes it copy-compatible.
        # HuggingFace's c_attn already concatenates query/key/value for all heads, matching
        # the fused layout directly, so no per-head splitting is needed.
        transformer_block.multi_head_attention.query_key_value_layer.weight.copy_(
            state_dict_huggingface[f"{huggingface_prefix}.attn.c_attn.weight"].t()
        )
        transformer_block.multi_head_attention.query_key_value_layer.bias.copy_(
            state_dict_huggingface[f"{huggingface_prefix}.attn.c_attn.bias"]
        )

        # Copy the weights and biases of the linear layer of the multi-head attention.
        transformer_block.multi_head_attention.linear.weight.copy_(
            state_dict_huggingface[f"{huggingface_prefix}.attn.c_proj.weight"].t()
        )
        transformer_block.multi_head_attention.linear.bias.copy_(
            state_dict_huggingface[f"{huggingface_prefix}.attn.c_proj.bias"]
        )

        # Copy the weights and biases of the feed-forward network's two linear layers.
        transformer_block.feed_forward.linear_1.weight.copy_(
            state_dict_huggingface[f"{huggingface_prefix}.mlp.c_fc.weight"].t()
        )
        transformer_block.feed_forward.linear_1.bias.copy_(
            state_dict_huggingface[f"{huggingface_prefix}.mlp.c_fc.bias"]
        )
        transformer_block.feed_forward.linear_2.weight.copy_(
            state_dict_huggingface[f"{huggingface_prefix}.mlp.c_proj.weight"].t()
        )
        transformer_block.feed_forward.linear_2.bias.copy_(
            state_dict_huggingface[f"{huggingface_prefix}.mlp.c_proj.bias"]
        )


class TransformerBlock(nn.Module):
    """One pre-norm transformer decoder block: causal self-attention plus a feed-forward network.

    Both sub-layers are wrapped in residual connections, each preceded by its own
    LayerNorm (GPT-2's "pre-norm" convention, as opposed to the original
    "post-norm" Transformer).
    """

    def __init__(self, config: GPTConfig) -> None:
        """Builds the two normalized, residual sub-layers that make up this block.

        Args:
            config: A validated GPTConfig describing the model's architecture.
        """
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(config.embedding_size)
        self.multi_head_attention = MultiHeadAttention(config)
        self.layer_norm_2 = nn.LayerNorm(config.embedding_size)
        self.feed_forward = FeedForward(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Applies the attention and feed-forward sub-layers, each with a residual connection.

        Args:
            hidden_states: Input of shape (B, S, E).

        Returns:
            Output of shape (B, S, E).
        """
        # Pre-norm: normalize before each sub-layer, then add its output back to the
        # un-normalized residual stream so gradients can flow through unimpeded.
        hidden_states = hidden_states + self.multi_head_attention(
            self.layer_norm_1(hidden_states)
        )
        hidden_states = hidden_states + self.feed_forward(
            self.layer_norm_2(hidden_states)
        )
        return hidden_states


class FeedForward(nn.Module):
    """The two-layer MLP sub-layer of a transformer block: expand, activate, project back down."""

    def __init__(self, config: GPTConfig) -> None:
        """Builds the expansion and projection layers, sized by `config.feed_forward_ratio`.

        Args:
            config: A validated GPTConfig describing the model's architecture.
        """
        super().__init__()

        # Gelu is used as per the original GPT-2 implementation, which uses the tanh approximation for speed.
        # Relu is not used to avoid the dead neuron problem whereby a neuron can get stuck outputting 0 and never recover.
        self.linear_1 = nn.Linear(
            config.embedding_size, config.feed_forward_ratio * config.embedding_size
        )
        self.gelu = nn.GELU(approximate="tanh")
        self.linear_2 = ResidualProjection(
            config.feed_forward_ratio * config.embedding_size, config.embedding_size
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Expands to the hidden feed-forward dimension, activates, then projects back to embedding_size.

        Args:
            hidden_states: Input of shape (B, S, E).

        Returns:
            Output of shape (B, S, E).
        """
        hidden_states = self.linear_1(hidden_states)  # B x S x E -> B x S x (ratio * E)
        hidden_states = self.gelu(hidden_states)  # B x S x (ratio * E)
        hidden_states = self.linear_2(hidden_states)  # B x S x (ratio * E) -> B x S x E
        return hidden_states


class MultiHeadAttention(nn.Module):
    """Fused multi-head causal self-attention: all heads computed in a single batched op.

    A single Linear projects to concatenated query/key/value for every head at
    once; each is reshaped to expose a separate head dimension so that
    `scaled_dot_product_attention` computes all heads in parallel.
    """

    def __init__(self, config: GPTConfig) -> None:
        """Builds the fused query/key/value projection and the output projection that merges heads.

        Args:
            config: A validated GPTConfig describing the model's architecture.
        """
        super().__init__()

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.attention_head_size

        # One Linear producing query, key, and value for all heads at once, instead of
        # separate Linears per head.
        # Output size is 3x the input size to hold concatenated query/key/value for every head.
        self.query_key_value_layer = nn.Linear(
            config.embedding_size, 3 * config.embedding_size
        )
        self.linear = ResidualProjection(config.embedding_size, config.embedding_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Runs fused multi-head causal attention and projects the merged output back to embedding_size.

        Args:
            hidden_states: Input of shape (B, S, E).

        Returns:
            Output of shape (B, S, E).
        """
        batch_size, sequence_size, embedding_size = hidden_states.size()

        query_key_value = self.query_key_value_layer(
            hidden_states
        )  # B x S x E -> B x S x 3E
        queries, keys, values = query_key_value.split(
            embedding_size, dim=-1
        )  # each B x S x E

        # Split the embedding dimension into (num_heads, head_size).
        queries = queries.reshape(
            batch_size,
            sequence_size,
            self.num_attention_heads,
            self.attention_head_size,
        )  # B x S x E -> B x S x num_heads x H
        # Move num_heads next to batch, so scaled_dot_product_attention treats each head as
        # an independent batch element and computes them all in one call.
        queries = queries.transpose(
            1, 2
        )  # B x S x num_heads x H -> B x num_heads x S x H

        # Do the same for keys.
        keys = keys.reshape(
            batch_size,
            sequence_size,
            self.num_attention_heads,
            self.attention_head_size,
        )  # B x S x E -> B x S x num_heads x H
        keys = keys.transpose(1, 2)  # B x S x num_heads x H -> B x num_heads x S x H

        # Do the same for values.
        values = values.reshape(
            batch_size,
            sequence_size,
            self.num_attention_heads,
            self.attention_head_size,
        )  # B x S x E -> B x S x num_heads x H
        values = values.transpose(
            1, 2
        )  # B x S x num_heads x H -> B x num_heads x S x H

        # Flash Attention; computes causal attention for every head in parallel.
        hidden_states = F.scaled_dot_product_attention(
            queries, keys, values, is_causal=True
        )  # B x num_heads x S x H

        # Move num_heads back next to sequence, undoing the earlier head-batching transpose.
        hidden_states = hidden_states.transpose(
            1, 2
        )  # B x num_heads x S x H -> B x S x num_heads x H
        # Merge num_heads and head_size back into a single embedding dimension.
        hidden_states = hidden_states.reshape(
            batch_size, sequence_size, embedding_size
        )  # B x S x num_heads x H -> B x S x E
        hidden_states = self.linear(hidden_states)  # B x S x E
        return hidden_states
