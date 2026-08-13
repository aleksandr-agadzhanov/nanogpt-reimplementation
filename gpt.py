import inspect
import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from transformers import GPT2LMHeadModel


class ResidualProjection(nn.Linear):
    """Linear layer at the end of a residual branch; scaled at init to control residual stream variance."""


@dataclass
class GPTConfig:
    vocabulary_size: int = 50304  # Optimization 5 - overridden to a nice number
    context_size: int = 1024
    embedding_size: int = 768
    attention_head_size: int = 64

    num_transformer_blocks: int = 12
    num_attention_heads: int = 12

    feed_forward_ratio: int = 4


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

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

        # Sharing weights between token embedding and language model head layers
        self.token_embedding.weight = self.language_model_head.weight

        self.register_buffer(
            "positions", torch.arange(config.context_size, dtype=torch.long)
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if isinstance(module, ResidualProjection):
                std = std / math.sqrt(2 * self.config.num_transformer_blocks)
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens, targets=None):
        _, sequence_size = tokens.size()  # B x S
        token_embeddings = self.token_embedding(tokens)  # B x S x E
        positions = self.positions.to(tokens.device)  # S
        positions = positions[:sequence_size]  # S
        position_embeddings = self.position_embedding(positions)  # B x S x E
        hidden_states = token_embeddings + position_embeddings  # B x S x E
        for transformer_block in self.transformer_blocks:
            hidden_states = transformer_block(hidden_states)  # B x S x E
        hidden_states = self.layer_norm(hidden_states)  # B x S x E
        logits = self.language_model_head(hidden_states)  # B x S x V

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, device):
        # Only decay 2D parameters. Decay weights in matmuls and embeddings, don't decay biases and layernorms.
        parameters = [parameter for _, parameter in self.named_parameters() if parameter.requires_grad]
        parameters_to_decay = [parameter for parameter in parameters if parameter.dim() >= 2]
        parameters_not_to_decay = [parameter for parameter in parameters if parameter.dim() < 2]
        parameter_groups = [
            {
                "params": parameters_to_decay,
                "weight_decay": weight_decay
            },
            {
                "params": parameters_not_to_decay,
                "weight_decay": 0.0
            }
        ]

        num_decayed_parameters = sum(parameter.numel() for parameter in parameters_to_decay)
        num_non_decayed_parameters = sum(parameter.numel() for parameter in parameters_not_to_decay)
        print(f"Decayed tensors - {len(parameters_to_decay)}, decayed parameters - {num_decayed_parameters:,}")
        print(f"Non-decayed tensors - {len(parameters_not_to_decay)}, non-decayed parameters - {num_non_decayed_parameters:,}")

        fused = "fused" in inspect.signature(torch.optim.AdamW).parameters and device == "cuda"
        print(f"Using fused Adam - {fused}")

        optimizer = torch.optim.AdamW(parameter_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=fused)
        return optimizer

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        print(f"loading weights from pretrained gpt: {model_type}")

        # Model dimensions mapped to this implementation's GPTConfig field names.
        config_args = {
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
        }[model_type]
        config_args["vocabulary_size"] = 50257  # always 50257 for GPT model checkpoints
        config_args["context_size"] = 1024  # always 1024 for GPT model checkpoints
        config_args["attention_head_size"] = (
            config_args["embedding_size"] // config_args["num_attention_heads"]
        )
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = cls(config)

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        with torch.no_grad():
            # Embeddings and final layer norm.
            model.token_embedding.weight.copy_(sd_hf["transformer.wte.weight"])
            model.position_embedding.weight.copy_(sd_hf["transformer.wpe.weight"])
            model.layer_norm.weight.copy_(sd_hf["transformer.ln_f.weight"])
            model.layer_norm.bias.copy_(sd_hf["transformer.ln_f.bias"])
            model.language_model_head.weight.copy_(sd_hf["lm_head.weight"])

            embedding_size = config.embedding_size
            head_size = config.attention_head_size

            for block_idx in range(config.num_transformer_blocks):
                hf_prefix = f"transformer.h.{block_idx}"
                block = model.transformer_blocks[block_idx]

                # Layer norms.
                block.layer_norm_1.weight.copy_(sd_hf[f"{hf_prefix}.ln_1.weight"])
                block.layer_norm_1.bias.copy_(sd_hf[f"{hf_prefix}.ln_1.bias"])
                block.layer_norm_2.weight.copy_(sd_hf[f"{hf_prefix}.ln_2.weight"])
                block.layer_norm_2.bias.copy_(sd_hf[f"{hf_prefix}.ln_2.bias"])

                # HF uses Conv1D layout for c_attn and c_proj, so transpose to Linear layout.
                qkv_weight = sd_hf[f"{hf_prefix}.attn.c_attn.weight"].t()  # (3E, E)
                q_weight = qkv_weight[:embedding_size, :]
                k_weight = qkv_weight[embedding_size : 2 * embedding_size, :]
                v_weight = qkv_weight[2 * embedding_size :, :]

                qkv_bias = sd_hf[f"{hf_prefix}.attn.c_attn.bias"]  # (3E,)
                q_bias = qkv_bias[:embedding_size]
                k_bias = qkv_bias[embedding_size : 2 * embedding_size]
                v_bias = qkv_bias[2 * embedding_size :]

                for head_idx, head in enumerate(
                    block.multi_head_attention.attention_heads
                ):
                    start = head_idx * head_size
                    end = start + head_size
                    head.query_layer.weight.copy_(q_weight[start:end, :])
                    head.query_layer.bias.copy_(q_bias[start:end])
                    head.key_layer.weight.copy_(k_weight[start:end, :])
                    head.key_layer.bias.copy_(k_bias[start:end])
                    head.value_layer.weight.copy_(v_weight[start:end, :])
                    head.value_layer.bias.copy_(v_bias[start:end])

                block.multi_head_attention.linear.weight.copy_(
                    sd_hf[f"{hf_prefix}.attn.c_proj.weight"].t()
                )
                block.multi_head_attention.linear.bias.copy_(
                    sd_hf[f"{hf_prefix}.attn.c_proj.bias"]
                )

                block.feed_forward.linear_1.weight.copy_(
                    sd_hf[f"{hf_prefix}.mlp.c_fc.weight"].t()
                )
                block.feed_forward.linear_1.bias.copy_(
                    sd_hf[f"{hf_prefix}.mlp.c_fc.bias"]
                )
                block.feed_forward.linear_2.weight.copy_(
                    sd_hf[f"{hf_prefix}.mlp.c_proj.weight"].t()
                )
                block.feed_forward.linear_2.bias.copy_(
                    sd_hf[f"{hf_prefix}.mlp.c_proj.bias"]
                )

        return model


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(config.embedding_size)
        self.multi_head_attention = MultiHeadAttention(config)
        self.layer_norm_2 = nn.LayerNorm(config.embedding_size)
        self.feed_forward = FeedForward(config)

    def forward(self, hidden_states):
        hidden_states = hidden_states + self.multi_head_attention(
            self.layer_norm_1(hidden_states)
        )
        hidden_states = hidden_states + self.feed_forward(
            self.layer_norm_2(hidden_states)
        )
        return hidden_states


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Gelu with tanh approximation because the original GPT-2 implementation used it.
        # Not Relu to avoid the dead Relu neuron problem.
        self.linear_1 = nn.Linear(
            config.embedding_size, config.feed_forward_ratio * config.embedding_size
        )
        self.gelu = nn.GELU(approximate="tanh")
        self.linear_2 = ResidualProjection(
            config.feed_forward_ratio * config.embedding_size, config.embedding_size
        )

    def forward(self, hidden_states):
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.gelu(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention_heads = nn.ModuleList(
            [AttentionHead(config) for _ in range(config.num_attention_heads)]
        )
        self.linear = ResidualProjection(config.embedding_size, config.embedding_size)

    def forward(self, hidden_states):
        hidden_states = [
            attention_head(hidden_states) for attention_head in self.attention_heads
        ]  # B x S x H each element
        hidden_states = torch.cat(hidden_states, -1)  # B x S x E
        hidden_states = self.linear(hidden_states)  # B x S x E
        return hidden_states


class AttentionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.key_layer = nn.Linear(config.embedding_size, config.attention_head_size)
        self.query_layer = nn.Linear(config.embedding_size, config.attention_head_size)
        self.value_layer = nn.Linear(config.embedding_size, config.attention_head_size)
        # self.register_buffer(
        #     "lower_triangular",
        #     torch.tril(torch.ones(config.context_size, config.context_size)),
        # )

    def forward(self, hidden_states):
        keys = self.key_layer(hidden_states)  # B x S x H
        queries = self.query_layer(hidden_states)  # B x S x H
        values = self.value_layer(hidden_states)  # B x S x H

        # What is happenning under the hood - not optimal
        # _, sequence_size, _ = hidden_states.size()  # B x S x E
        # scores = queries @ keys.transpose(
        #     -2, -1
        # )  # (B x S x H) @ (B x H x S) = B x S x S
        # scores_normalized = scores / math.sqrt(
        #     self.config.attention_head_size
        # )  # B x S x S
        # lower_triangular_resized = self.lower_triangular[
        #     :sequence_size, :sequence_size
        # ]  # S x S
        # scores_masked = scores_normalized.masked_fill(
        #     lower_triangular_resized == 0, float("-inf")
        # )  # S x S
        # scores_activated = scores_masked.softmax(-1)  # S x S
        # hidden_states = (
        #     scores_activated @ values
        # )  # (B x S x S) @ (B x S x H) = B x S x H

        # Optimization 4 - Flash Attention
        hidden_states = F.scaled_dot_product_attention(
            queries, keys, values, is_causal=True
        )

        return hidden_states
