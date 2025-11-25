# =============================================================
# reward_model.py — FINAL STABLE REWARD MODEL FOR RLHF
# Based on GPT2MiniXL backbone
# Supports: sequence scoring + ranking loss
# =============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.gpt2_mini_xl import GPT2MiniXL


class RewardModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        block_size=256,
        embed_dim=384,
        num_layers=12,
        num_heads=12,
        pooling="last"
    ):
        super().__init__()

        self.pooling = pooling

        # GPT backbone
        self.gpt = GPT2MiniXL(
            vocab_size=vocab_size,
            block_size=block_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
        )

        # Reward head (scalar output)
        self.reward_head = nn.Linear(embed_dim, 1)

    # ---------------------------------------------------------
    @torch.no_grad()
    def score(self, input_ids):
        """
        Compute scalar reward for each sequence.
        """
        logits, hidden = self.gpt(input_ids, return_hidden=True)

        if self.pooling == "last":
            rep = hidden[:, -1, :]         # final token embedding
        else:
            rep = hidden.mean(dim=1)       # mean pooling

        reward = self.reward_head(rep).squeeze(-1)
        return reward

    # ---------------------------------------------------------
    def forward(self, input_ids):
        """
        Usual forward: returns reward for each sequence.
        """
        logits, hidden = self.gpt(input_ids, return_hidden=True)

        if self.pooling == "last":
            rep = hidden[:, -1, :]
        else:
            rep = hidden.mean(dim=1)

        reward = self.reward_head(rep).squeeze(-1)
        return reward

    # ---------------------------------------------------------
    def ranking_loss(self, chosen_input, rejected_input, margin=0.0):
        """
        Pairwise preference loss:
        reward(chosen) > reward(rejected)
        """

        r_pos = self.forward(chosen_input)
        r_neg = self.forward(rejected_input)

        # Loss = -log(sigmoid(r_pos - r_neg))
        loss = -F.logsigmoid(r_pos - r_neg - margin).mean()

        return loss, r_pos, r_neg
