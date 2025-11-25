# ============================================================
# PPO Trainer — FINAL RLHF Version
# Clean, stable, sequence-level reward PPO
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils import clip_grad_norm_


class PPOTrainer:
    def __init__(
        self,
        actor_model,
        ref_model,
        reward_model,
        tokenizer,
        kl_coef=0.1,
        gamma=0.99,
        lam=0.95,
        clip_ratio=0.2,
        lr=1e-5,
        vf_lr=1e-5,
        entropy_coef=0.001,
        ppo_epochs=2,
        minibatch_size=4,
        device="cuda",
    ):

        self.actor = actor_model.to(device)
        self.ref = ref_model.to(device)
        self.reward_model = reward_model.to(device)
        self.tokenizer = tokenizer
        self.device = device

        # Freeze reference model and reward model
        for p in self.ref.parameters():
            p.requires_grad = False
        for p in self.reward_model.parameters():
            p.requires_grad = False

        # Independent critic head (sequence-level value estimation)
        embed_dim = actor_model.wte.weight.shape[1]
        self.critic = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, 1)
        ).to(device)

        # Optimizers
        self.actor_opt = torch.optim.AdamW(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.AdamW(self.critic.parameters(), lr=vf_lr)

        # PPO hyperparameters
        self.kl_coef = kl_coef
        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.ppo_epochs = ppo_epochs
        self.minibatch_size = minibatch_size

    # ---------------------------------------------------------
    # Padding helper
    # ---------------------------------------------------------
    def pad(self, seqs, pad_id=0):
        ml = max(len(s) for s in seqs)
        return [s + [pad_id] * (ml - len(s)) for s in seqs]

    # ---------------------------------------------------------
    # Generate responses from actor
    # ---------------------------------------------------------
    @torch.no_grad()
    def generate(self, prompts, max_new_tokens=64, top_k=50):

        encoded = []
        for p in prompts:
            ids = self.tokenizer.encode(p)
            ids = ids[-(self.actor.block_size - max_new_tokens):]
            encoded.append(ids)

        x = torch.tensor(self.pad(encoded), device=self.device)

        out = self.actor.generate(
            x, max_new_tokens=max_new_tokens, top_k=top_k
        )

        out = out.tolist()

        responses = []
        for i, prompt_ids in enumerate(encoded):
            resp = out[i][len(prompt_ids):]
            responses.append(resp)

        texts = [self.tokenizer.decode(r) for r in responses]

        return {
            "prompt_ids": encoded,
            "full_ids": out,
            "response_ids": responses,
            "responses": texts,
        }

    # ---------------------------------------------------------
    # Compute reward & KL
    # ---------------------------------------------------------
    @torch.no_grad()
    def compute_reward_kl(self, full_ids, prompt_ids):

        x = torch.tensor(self.pad(full_ids), device=self.device)

        # Sequence reward from Reward Model
        rm_scores = self.reward_model(x)       # shape: (B,)

        # Actor & Reference logits
        act_logits = self.actor(x)
        ref_logits = self.ref(x)

        act_lp = F.log_softmax(act_logits, dim=-1)
        ref_lp = F.log_softmax(ref_logits, dim=-1)

        # Logprob of *actual* sampled tokens
        act_taken = act_lp.gather(2, x.unsqueeze(-1)).squeeze(-1)
        ref_taken = ref_lp.gather(2, x.unsqueeze(-1)).squeeze(-1)

        kl = act_taken - ref_taken  # per-token KL

        # Make mask over response tokens only
        mask = torch.zeros_like(x, dtype=torch.float32)
        for i in range(len(full_ids)):
            p_len = len(prompt_ids[i])
            mask[i, p_len:len(full_ids[i])] = 1.0

        # Expand reward to each token position (sequence reward)
        rm_expanded = rm_scores.unsqueeze(1).expand_as(act_taken)

        rewards = (rm_expanded - self.kl_coef * kl) * mask

        return rewards, act_taken, mask

    # ---------------------------------------------------------
    # Critic forward (detached from actor)
    # ---------------------------------------------------------
    @torch.no_grad()
    def critic_values(self, x):
        logits, hidden = self.actor(x, return_hidden=True)
        rep = hidden[:, -1, :].detach()    # prevent gradient leak
        values = self.critic(rep).squeeze(-1)
        return values

    # ---------------------------------------------------------
    # GAE (Generalized Advantage Estimation)
    # ---------------------------------------------------------
    def gae(self, rewards, values, mask):
        B, T = rewards.shape
        advantages = torch.zeros_like(rewards, device=self.device)

        last_gae = torch.zeros(B, device=self.device)
        values_ext = torch.cat([values, torch.zeros(B, 1, device=self.device)], dim=1)

        for t in reversed(range(T)):
            m = mask[:, t]
            delta = rewards[:, t] + self.gamma * values_ext[:, t+1] - values_ext[:, t]
            delta = delta * m
            last_gae = delta + self.gamma * self.lam * last_gae
            advantages[:, t] = last_gae

        returns = advantages + values_ext[:, :-1]
        return advantages, returns

    # ---------------------------------------------------------
    # PPO Update Phase
    # ---------------------------------------------------------
    def ppo_update(self, batch):

        x = batch["input_ids"]
        old_lp = batch["old_logprobs"]
        adv = batch["advantages"]
        ret = batch["returns"]
        mask = batch["response_mask"]

        B, T = x.shape
        idxs = np.arange(B)

        stats = []

        for _ in range(self.ppo_epochs):
            np.random.shuffle(idxs)

            for s in range(0, B, self.minibatch_size):
                mb = idxs[s:s + self.minibatch_size]

                xs = x[mb]
                old = old_lp[mb]
                ad = adv[mb]
                re = ret[mb]
                mk = mask[mb]

                # ---------------------
                # ACTOR LOSS
                # ---------------------
                logits = self.actor(xs)
                lp = F.log_softmax(logits, dim=-1)
                new = lp.gather(2, xs.unsqueeze(-1)).squeeze(-1)

                ratio = torch.exp(new - old)

                unclipped = ratio * ad
                clipped = torch.clamp(
                    ratio, 
                    1 - self.clip_ratio, 
                    1 + self.clip_ratio
                ) * ad

                actor_loss = -(torch.min(unclipped, clipped) * mk).sum() / (mk.sum() + 1e-8)

                # Entropy bonus
                probs = torch.exp(lp)
                entropy = (-(probs * lp).sum(dim=-1) * mk).sum() / (mk.sum() + 1e-8)

                total_actor = actor_loss - self.entropy_coef * entropy

                self.actor_opt.zero_grad()
                total_actor.backward()
                clip_grad_norm_(self.actor.parameters(), 1.0)
                self.actor_opt.step()

                # ---------------------
                # CRITIC LOSS
                # ---------------------
                with torch.no_grad():
                    _, hidden = self.actor(xs, return_hidden=True)
                rep = hidden[:, -1, :].detach()

                values = self.critic(rep).squeeze(-1)
                target = re.sum(dim=1)  # sequence-level target

                critic_loss = F.mse_loss(values, target)

                self.critic_opt.zero_grad()
                critic_loss.backward()
                clip_grad_norm_(self.critic.parameters(), 1.0)
                self.critic_opt.step()

                stats.append(
                    (
                        total_actor.item(),
                        actor_loss.item(),
                        critic_loss.item(),
                        entropy.item(),
                    )
                )

        m = np.mean(stats, axis=0)

        return {
            "total_loss": m[0],
            "policy_loss": m[1],
            "value_loss": m[2],
            "entropy": m[3],
        }

    # ---------------------------------------------------------
    # One PPO training step on a batch of prompts
    # ---------------------------------------------------------
    def train_step(self, prompts):

        rollout = self.generate(prompts)
        full_ids = rollout["full_ids"]
        prompt_ids = rollout["prompt_ids"]

        rewards, old_lp, mask = self.compute_reward_kl(full_ids, prompt_ids)

        x = torch.tensor(self.pad(full_ids), device=self.device)

        values = self.critic_values(x).unsqueeze(1)
        values = values.repeat(1, x.shape[1])

        adv, rets = self.gae(rewards, values, mask)

        batch = {
            "input_ids": x,
            "old_logprobs": old_lp,
            "advantages": adv,
            "returns": rets,
            "response_mask": mask,
        }

        logs = self.ppo_update(batch)

        logs.update({
            "mean_reward": rewards.sum(dim=1).mean().item(),
            "mean_kl": (old_lp * mask).mean().item(),
            "mean_response_length": mask.sum(dim=1).float().mean().item(),
        })

        return logs, rollout["responses"]
