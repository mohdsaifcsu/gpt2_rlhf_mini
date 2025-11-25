# ================================================================
# train_ppo.py — FINAL RLHF PPO TRAINING SCRIPT
# Compatible with: GPT2MiniXL, RewardModel, PPOTrainer, GPT2BPETokenizer
# ================================================================

import os
import torch
from torch.utils.data import Dataset, DataLoader

from src.tokenizer.bpe_tokenizer import GPT2BPETokenizer
from src.models.gpt2_mini_xl import GPT2MiniXL
from src.rlhf.reward_model import RewardModel
from src.rlhf.ppo_trainer import PPOTrainer


# ================================================================
# Prompt Dataset (simple text per line)
# ================================================================
class PromptDataset(Dataset):
    def __init__(self, filepath):
        self.prompts = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                t = line.strip()
                if t:
                    self.prompts.append(t)

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]


# ================================================================
# PPO TRAINING LOOP
# ================================================================
def train_ppo():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Running PPO on: {device}")

    # ------------------------------------------------------------
    # Load tokenizer
    # ------------------------------------------------------------
    tokenizer = GPT2BPETokenizer(
        vocab_file="data/vocab/vocab.json",
        merges_file="data/vocab/merges.txt"
    )
    print("[INFO] Tokenizer loaded.")

    pad_id = tokenizer.pad_id

    # ------------------------------------------------------------
    # LOAD ACTOR MODEL (trainable)
    # ------------------------------------------------------------
    actor = GPT2MiniXL(
        vocab_size=tokenizer.vocab_size,
        block_size=256,
        embed_dim=384,
        num_layers=12,
        num_heads=12,
        dropout=0.1
    ).to(device)

    pretrained_path = "checkpoints/pretrained/gpt_xl_epoch1.pth"
    if os.path.exists(pretrained_path):
        print(f"[INFO] Loading pretrained Actor → {pretrained_path}")
        raw = torch.load(pretrained_path, map_location=device)
        if isinstance(raw, dict) and "state_dict" in raw:
            raw = raw["state_dict"]
        actor.load_state_dict(raw, strict=False)
    else:
        print("[WARN] No pretrained actor found. PPO will start from scratch.")

    # ------------------------------------------------------------
    # LOAD REFERENCE MODEL (frozen baseline)
    # ------------------------------------------------------------
    ref = GPT2MiniXL(
        vocab_size=tokenizer.vocab_size,
        block_size=256,
        embed_dim=384,
        num_layers=12,
        num_heads=12,
        dropout=0.1
    ).to(device)

    ref.load_state_dict(actor.state_dict(), strict=False)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad = False

    print("[INFO] Reference model ready (frozen).")

    # ------------------------------------------------------------
    # LOAD REWARD MODEL (frozen)
    # ------------------------------------------------------------
    reward_ckpt = "checkpoints/reward_model/rm_epoch1.pth"
    reward_model = RewardModel(
        vocab_size=tokenizer.vocab_size,
        block_size=256,
        embed_dim=384,
        num_layers=12,
        num_heads=12,
        pooling="last",
    ).to(device)

    if os.path.exists(reward_ckpt):
        print(f"[INFO] Loading Reward Model → {reward_ckpt}")
        raw = torch.load(reward_ckpt, map_location=device)
        if isinstance(raw, dict) and "state_dict" in raw:
            raw = raw["state_dict"]
        reward_model.load_state_dict(raw, strict=False)
    else:
        raise FileNotFoundError("Reward model checkpoint missing!")

    reward_model.eval()

    # ------------------------------------------------------------
    # CREATE PPO TRAINER
    # ------------------------------------------------------------
    trainer = PPOTrainer(
        actor_model=actor,
        ref_model=ref,
        reward_model=reward_model,
        tokenizer=tokenizer,
        kl_coef=0.1,
        gamma=0.99,
        lam=0.95,
        clip_ratio=0.2,
        lr=1e-5,
        vf_lr=1e-5,
        entropy_coef=0.001,
        ppo_epochs=2,
        minibatch_size=4,
        device=device,
    )

    # ------------------------------------------------------------
    # LOAD PROMPTS
    # ------------------------------------------------------------
    prompt_file = "data/raw/prompts.txt"
    dataset = PromptDataset(prompt_file)

    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    print(f"[INFO] Loaded {len(dataset)} PPO prompts.")

    # ------------------------------------------------------------
    # PPO MAIN LOOP
    # ------------------------------------------------------------
    epochs = 1
    steps_per_epoch = 50

    print("\n============= STARTING PPO TRAINING =============\n")

    for epoch in range(1, epochs + 1):
        print(f"\n================== EPOCH {epoch} ==================")

        for step, batch_prompts in enumerate(loader):

            if step >= steps_per_epoch:
                break

            prompts = list(batch_prompts)

            logs, responses = trainer.train_step(prompts)

            print(
                f"[E{epoch} | Step {step+1}/{steps_per_epoch}] "
                f"Loss={logs['total_loss']:.4f}  "
                f"Reward={logs['mean_reward']:.2f}  "
                f"KL={logs['mean_kl']:.3f}  "
                f"Len={logs['mean_response_length']:.1f}"
            )

            if (step + 1) % 10 == 0:
                print("\n--- Sample Responses ---")
                for r in responses[:2]:
                    print(" >", r[:300], "...\n")

        # SAVE CHECKPOINT
        os.makedirs("checkpoints/ppo/", exist_ok=True)
        ckpt_path = f"checkpoints/ppo/actor_epoch{epoch}.pth"
        torch.save(actor.state_dict(), ckpt_path)

        print(f"[INFO] Saved PPO Actor → {ckpt_path}")


# =============================================================
if __name__ == "__main__":
    train_ppo()
