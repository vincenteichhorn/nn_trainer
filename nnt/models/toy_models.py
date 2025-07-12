from types import SimpleNamespace
import torch

import torch.nn as nn
import torch.nn.functional as F


class ToyClassificationModel(nn.Module):
    def __init__(self, input_size=10, hidden_size=20, output_size=2):
        super(ToyClassificationModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        loss = self.loss_fn(x, y) if y is not None else None
        return SimpleNamespace(loss=loss, logits=x)


class ToyLanguageModel(nn.Module):
    def __init__(self, vocab_size=100, embed_dim=32, num_heads=2, num_layers=2, hidden_dim=64, max_seq_len=128):
        super(ToyLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.transformer_blocks = nn.ModuleList(
            [ToyTransformerBlock(embed_dim, num_heads, hidden_dim) for _ in range(num_layers)]
        )
        self.fc = nn.Linear(embed_dim, vocab_size)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, input_ids, labels=None, attention_mask=None):
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

    def generate(self, input_ids, max_length=20, temperature=1.0, eot_token_id=None):
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
    def __init__(self, embed_dim, num_heads, hidden_dim):
        super().__init__()
        self.attn = ToyMultiHeadSelfAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(nn.Linear(embed_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, embed_dim))
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, attn_mask=None):
        attn_out = self.attn(x, attn_mask)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x


class ToyMultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, attn_mask=None):
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
