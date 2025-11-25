# pretrain_gpt.py – FINAL FIXED VERSION
# Uses full sliding-window dataset (maximizes dataset size)

import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import os

from src.models.gpt2_mini_xl import GPT2MiniXL
from src.tokenizer.bpe_tokenizer import GPT2BPETokenizer


# ============================================================
# Sliding Window Dataset (Correct version)
# ============================================================
class SlidingWindowDataset(Dataset):
    def __init__(self, text_ids, block_size=256, stride=64):
        self.block_size = block_size
        self.stride = stride

        self.samples = []
        for i in range(0, len(text_ids) - block_size - 1, stride):
            x = text_ids[i : i + block_size]
            y = text_ids[i + 1 : i + block_size + 1]
            self.samples.append((x, y))

        print(f"Total usable chunks (sliding window): {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x), torch.tensor(y)


# ============================================================
# Training Function
# ============================================================
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Tokenizer
    tokenizer = GPT2BPETokenizer(
        vocab_file="data/vocab/vocab.json",
        merges_file="data/vocab/merges.txt"
    )

    # Load text
    text_path = "data/raw/pretrain_corpus.txt"
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()

    print("Tokenizing corpus (this will take a moment)...")
    text_ids = tokenizer.encode(text, add_special_tokens=True)

    # Dataset (sliding window)
    dataset = SlidingWindowDataset(
        text_ids=text_ids,
        block_size=256,
        stride=64       # <—— Key to producing 200k+ samples
    )

    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Model
    model = GPT2MiniXL(
        vocab_size=tokenizer.vocab_size,
        block_size=256,
        embed_dim=384,
        num_layers=12,
        num_heads=12
    ).to(device)

    optimz = optim.AdamW(model.parameters(), lr=3e-4)

    print("Starting GPT-XL pretraining...")

    epochs = 1   # increase if needed
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for step, (x, y) in enumerate(loader, start=1):
            x, y = x.to(device), y.to(device)

            logits, loss = model(x, y)

            optimz.zero_grad()
            loss.backward()
            optimz.step()

            running_loss += loss.item()

            if step % 200 == 0:
                print(f"[Epoch {epoch+1}] Step {step} Loss: {loss.item():.4f}")

        avg_loss = running_loss / len(loader)
        print(f"Epoch {epoch+1} finished. Avg Loss = {avg_loss:.4f}")

        os.makedirs("checkpoints/pretrained", exist_ok=True)
        torch.save(model.state_dict(), f"checkpoints/pretrained/gpt_xl_epoch{epoch+1}.pth")
        print(f"Saved checkpoint for epoch {epoch+1}")

    print("Pretraining complete!")


if __name__ == "__main__":
    train()
