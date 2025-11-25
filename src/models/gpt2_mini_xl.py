# ============================================================
# gpt2_mini_xl.py — FINAL STABLE GPT MODEL
# 12 layers, 384-dim, 12 heads
# Fully RLHF-compatible (SFT, RM, PPO)
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm


# ============================================================
# Multi-Head Attention
# ============================================================
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, block_size, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.block_size = block_size

        self.c_attn = nn.Linear(embed_dim, 3 * embed_dim)
        self.c_proj = nn.Linear(embed_dim, embed_dim)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Causal mask
        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("mask", mask)

    def forward(self, x):
        B, T, C = x.shape

        # Compute Q, K, V
        qkv = self.c_attn(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        att = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        causal = self.mask[:T, :T]
        att = att.masked_fill(causal == 0, float("-inf"))

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        out = self.c_proj(out)
        return self.resid_dropout(out)


# ============================================================
# Feed-Forward Network
# ============================================================
class FeedForward(nn.Module):
    def __init__(self, embed_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, 4 * embed_dim)
        self.fc2 = nn.Linear(4 * embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return self.dropout(x)


# ============================================================
# Transformer Block
# ============================================================
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, block_size, dropout=0.1):
        super().__init__()
        self.ln1 = LayerNorm(embed_dim)
        self.ln2 = LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, block_size, dropout)
        self.ff = FeedForward(embed_dim, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


# ============================================================
# GPT-2 MINI XL MODEL
# ============================================================
class GPT2MiniXL(nn.Module):
    def __init__(
        self,
        vocab_size,
        block_size=256,
        embed_dim=384,
        num_layers=12,
        num_heads=12,
        dropout=0.1,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.block_size = block_size

        self.wte = nn.Embedding(vocab_size, embed_dim)
        self.wpe = nn.Embedding(block_size, embed_dim)

        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, block_size, dropout)
            for _ in range(num_layers)
        ])

        self.ln_f = LayerNorm(embed_dim)

        # LM Head (tied weights)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight

        self.apply(self._init_weights)

    # --------------------------------------------------------
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # --------------------------------------------------------
    def forward(self, idx, targets=None, return_hidden=False):
        B, T = idx.shape
        assert T <= self.block_size

        pos = torch.arange(0, T, device=idx.device)

        x = self.wte(idx) + self.wpe(pos)
        x = self.drop(x)

        last_hidden = None

        for block in self.blocks:
            x = block(x)
            last_hidden = x

        x = self.ln_f(x)
        logits = self.lm_head(x)

        if return_hidden:
            return logits, last_hidden

        if targets is None:
            return logits

        loss = F.cross_entropy(
            logits.view(-1, self.vocab_size),
            targets.view(-1),
            ignore_index=-100,
        )

        return logits, loss

    # --------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens=64,
        temperature=1.0,
        top_k=None
    ):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]

            logits = self(idx_cond)[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                cutoff = v[:, -1].unsqueeze(1)
                logits = torch.where(logits < cutoff, -1e10, logits)

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)

            idx = torch.cat([idx, next_id], dim=1)

        return idx
