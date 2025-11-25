# =============================================================
# train_reward_model.py
# =============================================================

import json
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import os
from tqdm import tqdm

from src.tokenizer.bpe_tokenizer import GPT2BPETokenizer
from src.rlhf.reward_model import RewardModel


# ============================================================
# Preference Dataset
# ============================================================
class PreferenceDataset(Dataset):
    def __init__(self, filepath, tokenizer, block_size=256):
        self.samples = []

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))

        self.tokenizer = tokenizer
        self.block_size = block_size
        self.pad_id = tokenizer.pad_id

    def encode(self, text):
        ids = self.tokenizer.encode(text)
        if len(ids) > self.block_size:
            ids = ids[-self.block_size:]   # Keep right most
        return ids

    def pad(self, ids):
        return ids + [self.pad_id] * (self.block_size - len(ids))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        prompt = item["prompt"]
        chosen = item["chosen"]
        rejected = item["rejected"]

        chosen_ids = self.encode(prompt + " " + chosen)
        rejected_ids = self.encode(prompt + " " + rejected)

        return {
            "chosen": torch.tensor(self.pad(chosen_ids), dtype=torch.long),
            "rejected": torch.tensor(self.pad(rejected_ids), dtype=torch.long)
        }


# ============================================================
# Train Reward Model
# ============================================================
def train_reward_model():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Training Reward Model on {device}")

    # ----------------------------------------------------------
    # Tokenizer
    # ----------------------------------------------------------
    tokenizer = GPT2BPETokenizer(
        vocab_file="data/vocab/vocab.json",
        merges_file="data/vocab/merges.txt"
    )

    # ----------------------------------------------------------
    # Dataset
    # ----------------------------------------------------------
    dataset_path = "data/preferences/pairs.jsonl"
    dataset = PreferenceDataset(dataset_path, tokenizer)

    loader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True)

    print(f"[INFO] Loaded {len(dataset)} preference pairs.")

    # ----------------------------------------------------------
    # Reward Model
    # ----------------------------------------------------------
    rm = RewardModel(
        vocab_size=tokenizer.vocab_size,
        block_size=256,
        embed_dim=384,
        num_layers=12,
        num_heads=12,
        pooling="last"        # sequence level reward
    ).to(device)

    # Freeze GPT weights - ONLY reward head trains
    for p in rm.gpt.parameters():
        p.requires_grad = False

    # ----------------------------------------------------------
    # Load pretrained GPT weights
    # ----------------------------------------------------------
    pretrained_path = "checkpoints/pretrained/gpt_xl_epoch1.pth"
    if os.path.exists(pretrained_path):
        print(f"[INFO] Loading pretrained GPT → {pretrained_path}")
        state = torch.load(pretrained_path, map_location=device)
        if "state_dict" in state:
            state = state["state_dict"]
        rm.gpt.load_state_dict(state, strict=False)

    # ----------------------------------------------------------
    # Optimizer
    # ----------------------------------------------------------
    optimizer = optim.AdamW(
        [p for p in rm.parameters() if p.requires_grad],
        lr=1e-5
    )

    # ----------------------------------------------------------
    # Training Loop
    # ----------------------------------------------------------
    epochs = 1
    rm.train()

    print("\n========== TRAINING REWARD MODEL ==========\n")

    for epoch in range(epochs):
        running_loss = 0.0

        for batch in tqdm(loader, desc=f"Epoch {epoch+1}"):

            chosen = batch["chosen"].to(device)
            rejected = batch["rejected"].to(device)

            loss, r_pos, r_neg = rm.ranking_loss(
                chosen_input=chosen,
                rejected_input=rejected,
                margin=0.0
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rm.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(loader)
        print(f"[INFO] Epoch {epoch+1} - Avg Reward Loss = {avg_loss:.4f}")

        save_path = f"checkpoints/reward_model/rm_epoch{epoch+1}.pth"
        os.makedirs("checkpoints/reward_model", exist_ok=True)
        torch.save({"state_dict": rm.state_dict()}, save_path)

        print(f"[INFO] Saved Reward Model -> {save_path}\n")

    print("Reward Model training complete!\n")


if __name__ == "__main__":
    train_reward_model()
