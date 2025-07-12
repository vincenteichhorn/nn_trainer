from types import SimpleNamespace
import torch

import torch.nn as nn
import torch.nn.functional as F


class ToyClassificationModel(nn.Module):
    """
    Simple feedforward neural network for classification tasks.

    Args:
        input_size (int): Number of input features.
        hidden_size (int): Number of hidden units.
        output_size (int): Number of output classes.

    Example:
        model = ToyClassificationModel(input_size=10, hidden_size=20, output_size=2)
        x = torch.randn(5, 10)
        y = torch.randint(0, 2, (5,))
        out = model(x, y)
        print(out.loss, out.logits)
    """

    def __init__(self, input_size: int = 10, hidden_size: int = 20, output_size: int = 2):
        """
        Initialize the ToyClassificationModel.
        """
        super(ToyClassificationModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor, y: torch.Tensor = None) -> SimpleNamespace:
        """
        Forward pass for the classification model.

        Args:
            x (Tensor): Input features.
            y (Tensor, optional): Target labels.
        Returns:
            SimpleNamespace: Contains 'loss' and 'logits'.
        """
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        loss = self.loss_fn(x, y) if y is not None else None
        return SimpleNamespace(loss=loss, logits=x)


class ToyLanguageModel(nn.Module):
    """
    Simple transformer-based language model for sequence modeling and generation.

    Args:
        vocab_size (int): Vocabulary size.
        embed_dim (int): Embedding dimension.
        num_heads (int): Number of attention heads.
        num_layers (int): Number of transformer blocks.
        hidden_dim (int): Hidden dimension in transformer blocks.
        max_seq_len (int): Maximum sequence length.

    Example:
        model = ToyLanguageModel(vocab_size=100)
        input_ids = torch.randint(0, 100, (2, 10))
        out = model(input_ids)
        print(out.logits.shape)
        generated = model.generate(input_ids, max_length=5)
        print(generated)
    """

    def __init__(
        self,
        vocab_size: int = 100,
        embed_dim: int = 32,
        num_heads: int = 2,
        num_layers: int = 2,
        hidden_dim: int = 64,
        max_seq_len: int = 128,
    ):
        """
        Initialize the ToyLanguageModel.
        """
        super(ToyLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.transformer_blocks = nn.ModuleList(
            [ToyTransformerBlock(embed_dim, num_heads, hidden_dim) for _ in range(num_layers)]
        )
        self.fc = nn.Linear(embed_dim, vocab_size)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
    ) -> SimpleNamespace:
        """
        Forward pass for the language model.

        Args:
            input_ids (Tensor): Input token IDs.
            labels (Tensor, optional): Target token IDs for loss computation.
            attention_mask (Tensor, optional): Attention mask for padding.
        Returns:
            SimpleNamespace: Contains 'loss' and 'logits'.
        """
        seq_len = input_ids.size(1)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        safe_input_ids = input_ids.clamp(0, self.embedding.num_embeddings - 1)
        x = self.embedding(safe_input_ids) + self.position_embedding(positions)

        if attention_mask is not None:
            attn_mask = attention_mask == 0
        else:
            attn_mask = None

        for block in self.transformer_blocks:
            x = block(x, attn_mask)

        logits = self.fc(x)
        if labels is not None:
            # compute loss only if labels are provided
            loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

        else:
            loss = None
        return SimpleNamespace(loss=loss, logits=logits)

    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 20,
        temperature: float = 1.0,
        eot_token_id: int = None,
    ) -> torch.Tensor:
        """
        Generate sequences from input_ids using autoregressive sampling.

        Args:
            input_ids (Tensor): Initial input token IDs.
            max_length (int): Maximum length of generated sequence.
            temperature (float): Sampling temperature.
            eot_token_id (int, optional): End-of-text token ID to stop generation.
        Returns:
            Tensor: Generated token IDs.
        """
        self.eval()
        generated = input_ids
        for _ in range(max_length):
            # Use forward to get logits
            out = self.forward(generated)
            logits = out.logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
            if eot_token_id is not None and next_token.item() == eot_token_id:
                break
        return generated


class ToyTransformerBlock(nn.Module):
    """
    Transformer block with multi-head self-attention and feedforward layers.

    Args:
        embed_dim (int): Embedding dimension.
        num_heads (int): Number of attention heads.
        hidden_dim (int): Hidden dimension in feedforward layers.
    """

    def __init__(self, embed_dim: int, num_heads: int, hidden_dim: int):
        """
        Initialize the ToyTransformerBlock.
        """
        super().__init__()
        self.attn = ToyMultiHeadSelfAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(nn.Linear(embed_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, embed_dim))
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for the transformer block.

        Args:
            x (Tensor): Input tensor.
            attn_mask (Tensor, optional): Attention mask.
        Returns:
            Tensor: Output tensor after attention and feedforward layers.
        """
        attn_out = self.attn(x, attn_mask)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x


class ToyMultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention layer for transformer blocks.

    Args:
        embed_dim (int): Embedding dimension.
        num_heads (int): Number of attention heads.
    """

    def __init__(self, embed_dim: int, num_heads: int):
        """
        Initialize the ToyMultiHeadSelfAttention layer.
        """
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for multi-head self-attention.

        Args:
            x (Tensor): Input tensor of shape (batch, seq_len, embed_dim).
            attn_mask (Tensor, optional): Attention mask.
        Returns:
            Tensor: Output tensor after attention.
        """
        batch_size, seq_len, embed_dim = x.size()
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)
            attn_scores = attn_scores.masked_fill(attn_mask, float("-inf"))
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        return self.out_proj(attn_output)
