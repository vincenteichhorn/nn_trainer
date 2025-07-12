import torch
import pytest
from nnt.models.toy_models import ToyClassificationModel, ToyLanguageModel
from nnt.models.toy_models import ToyTransformerBlock
from nnt.models.toy_models import ToyMultiHeadSelfAttention
from nnt.models.toy_models import ToyMultiHeadSelfAttention


def test_toy_classification_model_forward_no_labels():
    model = ToyClassificationModel(input_size=10, hidden_size=20, output_size=2)
    x = torch.randn(5, 10)
    output = model(x)
    assert hasattr(output, "logits")
    assert output.loss is None
    assert output.logits.shape == (5, 2)


def test_toy_classification_model_forward_with_labels():
    model = ToyClassificationModel(input_size=10, hidden_size=20, output_size=2)
    x = torch.randn(5, 10)
    y = torch.randint(0, 2, (5,))
    output = model(x, y)
    assert hasattr(output, "loss")
    assert output.loss is not None
    assert output.logits.shape == (5, 2)
    assert output.loss.shape == ()


def test_toy_classification_model_different_batch_sizes():
    model = ToyClassificationModel(input_size=10, hidden_size=20, output_size=2)
    for batch_size in [1, 10, 100]:
        x = torch.randn(batch_size, 10)
        y = torch.randint(0, 2, (batch_size,))
        output = model(x, y)
        assert output.logits.shape == (batch_size, 2)
        assert output.loss is not None


def test_toy_language_model_forward_no_labels():
    model = ToyLanguageModel(vocab_size=50, embed_dim=16, num_heads=2, num_layers=2, hidden_dim=32, max_seq_len=20)
    input_ids = torch.randint(0, 50, (4, 20))
    output = model(input_ids)
    assert hasattr(output, "logits")
    assert output.loss is None
    assert output.logits.shape == (4, 20, 50)


def test_toy_language_model_forward_with_labels():
    model = ToyLanguageModel(vocab_size=50, embed_dim=16, num_heads=2, num_layers=2, hidden_dim=32, max_seq_len=20)
    input_ids = torch.randint(0, 50, (4, 20))
    labels = torch.randint(0, 50, (4, 20))
    output = model(input_ids, labels)
    assert hasattr(output, "loss")
    assert output.loss is not None
    assert output.logits.shape == (4, 20, 50)
    assert output.loss.shape == ()


def test_toy_language_model_with_attention_mask():
    model = ToyLanguageModel(vocab_size=30, embed_dim=8, num_heads=2, num_layers=1, hidden_dim=16, max_seq_len=10)
    input_ids = torch.randint(0, 30, (2, 10))
    labels = torch.randint(0, 30, (2, 10))
    attention_mask = torch.ones(2, 10)
    attention_mask[:, -3:] = 0  # mask last 3 tokens
    output = model(input_ids, labels, attention_mask)
    assert output.loss is not None
    assert output.logits.shape == (2, 10, 30)


def test_toy_transformer_block_output_shape():
    block = ToyTransformerBlock(embed_dim=12, num_heads=2, hidden_dim=24)
    x = torch.randn(3, 7, 12)
    out = block(x)
    assert out.shape == (3, 7, 12)


def test_toy_multihead_self_attention_output_shape():
    attn = ToyMultiHeadSelfAttention(embed_dim=16, num_heads=4)
    x = torch.randn(2, 5, 16)
    out = attn(x)
    assert out.shape == (2, 5, 16)


def test_toy_multihead_self_attention_with_mask():
    attn = ToyMultiHeadSelfAttention(embed_dim=8, num_heads=2)
    x = torch.randn(1, 4, 8)
    attn_mask = torch.tensor([[1, 0, 0, 1]], dtype=torch.bool)  # mask some tokens
    out = attn(x, attn_mask)
    assert out.shape == (1, 4, 8)


def test_toy_language_model_generate():
    model = ToyLanguageModel(vocab_size=50, embed_dim=16, num_heads=2, num_layers=2, hidden_dim=32, max_seq_len=20)
    input_ids = torch.randint(0, 50, (1, 10))  # batch size of 1
    generated = model.generate(input_ids, max_length=5)
    assert generated.shape == (1, 15)  # original length + max_length
    assert generated.dtype == torch.int64
    assert torch.all(generated < 50)  # all tokens should be within vocab size
