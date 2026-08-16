import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gpt import GPTConfig


def test_default_config_is_valid():
    config = GPTConfig()
    assert config.embedding_size == config.attention_head_size * config.num_attention_heads


def test_custom_valid_config_is_accepted():
    config = GPTConfig(
        vocabulary_size=32768,
        context_size=2048,
        embedding_size=1024,
        attention_head_size=64,
        num_attention_heads=16,
        num_transformer_blocks=24,
        feed_forward_ratio=4,
    )
    assert config.embedding_size == 1024
    assert config.num_transformer_blocks == 24


def test_embedding_size_need_not_be_a_power_of_two():
    # embedding_size is derived from head_size * num_heads, so it isn't itself
    # required to be a power of 2 (e.g. GPT-2's 768 = 64 * 12).
    config = GPTConfig(embedding_size=768, attention_head_size=64, num_attention_heads=12)
    assert config.embedding_size == 768


def test_vocabulary_size_need_not_be_a_power_of_two():
    # Real tokenizer vocabularies (e.g. GPT-2's 50257) aren't powers of 2.
    config = GPTConfig(vocabulary_size=50257)
    assert config.vocabulary_size == 50257


@pytest.mark.parametrize(
    "field_name",
    [
        "vocabulary_size",
        "context_size",
        "embedding_size",
        "attention_head_size",
        "num_transformer_blocks",
        "num_attention_heads",
        "feed_forward_ratio",
    ],
)
@pytest.mark.parametrize("invalid_value", [0, -1])
def test_non_positive_fields_are_rejected(field_name, invalid_value):
    with pytest.raises(ValueError, match=f"{field_name} must be positive"):
        GPTConfig(**{field_name: invalid_value})


def test_embedding_size_mismatch_is_rejected():
    with pytest.raises(ValueError, match="embedding_size must equal"):
        GPTConfig(embedding_size=768, attention_head_size=64, num_attention_heads=10)


@pytest.mark.parametrize(
    "field_name, invalid_value, extra_kwargs",
    [
        ("context_size", 1000, {}),
        # embedding_size must stay consistent with head_size * num_heads so the
        # equality check passes and the power-of-2 check is what actually fires.
        ("attention_head_size", 63, {"embedding_size": 63 * 12}),
        ("feed_forward_ratio", 3, {}),
    ],
)
def test_non_power_of_two_fields_are_rejected(field_name, invalid_value, extra_kwargs):
    with pytest.raises(ValueError, match=f"{field_name} must be a power of 2"):
        GPTConfig(**{field_name: invalid_value, **extra_kwargs})
