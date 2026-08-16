import math
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gpt import (
    GPT,
    FeedForward,
    GPTConfig,
    MultiHeadAttention,
    ResidualProjection,
    TransformerBlock,
)


@pytest.fixture
def small_config() -> GPTConfig:
    """A small, fast-to-construct GPTConfig satisfying all validation rules."""
    return GPTConfig(
        vocabulary_size=100,
        context_size=16,
        embedding_size=32,
        attention_head_size=8,
        num_attention_heads=4,
        num_transformer_blocks=2,
        feed_forward_ratio=4,
    )


# --- ResidualProjection ---


def test_residual_projection_behaves_like_a_linear_layer():
    projection = ResidualProjection(8, 4)
    assert isinstance(projection, nn.Linear)
    output = projection(torch.randn(2, 8))
    assert output.shape == (2, 4)


# --- FeedForward ---


def test_feed_forward_expands_then_projects_back(small_config):
    feed_forward = FeedForward(small_config)
    assert feed_forward.linear_1.out_features == (
        small_config.feed_forward_ratio * small_config.embedding_size
    )
    assert isinstance(feed_forward.linear_2, ResidualProjection)

    hidden_states = torch.randn(2, 6, small_config.embedding_size)
    output = feed_forward(hidden_states)
    assert output.shape == hidden_states.shape


# --- MultiHeadAttention ---


def test_multi_head_attention_projects_to_and_from_embedding_size(small_config):
    attention = MultiHeadAttention(small_config)
    assert (
        attention.query_key_value_layer.out_features == 3 * small_config.embedding_size
    )
    assert isinstance(attention.linear, ResidualProjection)

    hidden_states = torch.randn(2, 6, small_config.embedding_size)
    output = attention(hidden_states)
    assert output.shape == hidden_states.shape


def test_multi_head_attention_is_causal(small_config):
    # Changing a later position's input must not change any earlier position's output.
    attention = MultiHeadAttention(small_config)
    attention.eval()

    hidden_states = torch.randn(2, 6, small_config.embedding_size)
    modified_hidden_states = hidden_states.clone()
    modified_hidden_states[:, -1, :] = torch.randn(2, small_config.embedding_size)

    with torch.no_grad():
        output = attention(hidden_states)
        modified_output = attention(modified_hidden_states)

    assert torch.allclose(output[:, :-1], modified_output[:, :-1], atol=1e-6)


# --- TransformerBlock ---


def test_transformer_block_preserves_shape(small_config):
    block = TransformerBlock(small_config)
    hidden_states = torch.randn(2, 6, small_config.embedding_size)
    output = block(hidden_states)
    assert output.shape == hidden_states.shape


# --- GPT ---


def test_gpt_ties_token_embedding_and_lm_head_weights(small_config):
    model = GPT(small_config)
    assert model.token_embedding.weight is model.language_model_head.weight


def test_gpt_forward_without_targets_returns_no_loss(small_config):
    model = GPT(small_config)
    tokens = torch.randint(0, small_config.vocabulary_size, (2, 6))

    logits, loss = model(tokens)

    assert logits.shape == (2, 6, small_config.vocabulary_size)
    assert loss is None


def test_gpt_forward_with_targets_computes_scalar_loss(small_config):
    model = GPT(small_config)
    tokens = torch.randint(0, small_config.vocabulary_size, (2, 6))
    targets = torch.randint(0, small_config.vocabulary_size, (2, 6))

    logits, loss = model(tokens, targets)

    assert logits.shape == (2, 6, small_config.vocabulary_size)
    assert loss is not None
    assert loss.dim() == 0


def test_gpt_forward_raises_when_sequence_exceeds_context_size(small_config):
    model = GPT(small_config)
    tokens = torch.randint(
        0, small_config.vocabulary_size, (1, small_config.context_size + 1)
    )

    with pytest.raises(ValueError, match="exceeds context_size"):
        model(tokens)


def test_init_weights_zeroes_linear_bias(small_config):
    model = GPT(small_config)
    linear = nn.Linear(16, 16)

    model._init_weights(linear)

    assert torch.all(linear.bias == 0)


def test_init_weights_scales_residual_projections_down(small_config):
    # ResidualProjection layers should be initialized with a smaller std than plain
    # Linear layers, scaled by 1 / sqrt(2 * num_transformer_blocks).
    torch.manual_seed(0)
    model = GPT(small_config)

    plain_linear = nn.Linear(2048, 2048)
    model._init_weights(plain_linear)
    residual_projection = ResidualProjection(2048, 2048)
    model._init_weights(residual_projection)

    expected_ratio = 1 / math.sqrt(2 * small_config.num_transformer_blocks)
    actual_ratio = (
        residual_projection.weight.std().item() / plain_linear.weight.std().item()
    )
    assert actual_ratio == pytest.approx(expected_ratio, rel=0.1)


def test_get_configured_optimizer_splits_decay_and_non_decay_params(small_config):
    model = GPT(small_config)

    optimizer = model.get_configured_optimizer(
        weight_decay=0.1,
        learning_rate=1e-3,
        device="cpu",
        is_master_process=False,
    )

    assert isinstance(optimizer, torch.optim.AdamW)
    decay_group, no_decay_group = optimizer.param_groups
    assert decay_group["weight_decay"] == 0.1
    assert no_decay_group["weight_decay"] == 0.0
    assert all(parameter.dim() >= 2 for parameter in decay_group["params"])
    assert all(parameter.dim() < 2 for parameter in no_decay_group["params"])


def test_from_pretrained_rejects_unknown_model_type():
    # Validated before any network access, so this doesn't require downloading weights.
    with pytest.raises(ValueError, match="Unknown model_type"):
        GPT.from_pretrained("not-a-real-gpt2-checkpoint")
