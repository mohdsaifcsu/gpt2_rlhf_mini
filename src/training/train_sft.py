# ===============================================================
# train_sft.py — FINAL STABLE SFT TRAINER FOR RLHF PIPELINE
# ===============================================================

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from torch.amp import autocast, GradScaler

from src.models.gpt2_mini_xl import GPT2MiniXL
from src.tokenizer.bpe_tokenizer import GPT2BPETokenizer


# ===============================================================
# SFT Dataset
# ===============================================================
class SFTDataset(Dataset):
    def __init__(self, path, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.max_len = max_len

        with open(path, "r", encoding="utf-8") as f:
            self.data = [json.loads(x) for x in f]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        prompt = ex["prompt"].strip()
        response = ex["response"].strip()

        # Standard SFT formatting
        text = f"Instruction: {prompt}\nResponse: {response}"

        ids = self.tokenizer.encode(text)       # BOS ... EOS
        ids = ids[: self.max_len]               # truncate

        return torch.tensor(ids, dtype=torch.long)


# ===============================================================
# Collate Function
# ===============================================================
def collate_fn(batch):
    batch = pad_sequence(batch, batch_first=True, padding_value=0)
    labels = batch.clone()
    labels[labels == 0] = -100     # Ignore PAD in loss
    return batch, labels


# ===============================================================
# Train SFT
# ===============================================================
def train_sft():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # -----------------------------------------------------------
    # Paths
    # -----------------------------------------------------------
    DATA_PATH = "data/sft/sft_dataset.jsonl"
    SAVE_DIR = "checkpoints/sft"
    VOCAB_PATH = "data/vocab/vocab.json"
    MERGES_PATH = "data/vocab/merges.txt"
    os.makedirs(SAVE_DIR, exist_ok=True)

    # -----------------------------------------------------------
    # Hyperparameters
    # -----------------------------------------------------------
    epochs = 3
    batch_size = 2
    grad_accum = 12
    lr = 1.5e-5
    warmup_steps = 150
    max_len = 512

    # -----------------------------------------------------------
    # Load Tokenizer
    # -----------------------------------------------------------
    tokenizer = GPT2BPETokenizer(
        vocab_file=VOCAB_PATH,
        merges_file=MERGES_PATH
    )
    print("[INFO] Tokenizer loaded.")

    # -----------------------------------------------------------
    # Dataset + Loader
    # -----------------------------------------------------------
    dataset = SFTDataset(DATA_PATH, tokenizer, max_len=max_len)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True
    )
    print(f"[INFO] Loaded SFT dataset → {len(dataset)} examples")

    # -----------------------------------------------------------
    # Model
    # -----------------------------------------------------------
    model = GPT2MiniXL(vocab_size=tokenizer.vocab_size)
    model = model.to(device)
    model.train()
    print("[INFO] Model loaded.")

    # -----------------------------------------------------------
    # Optimizer + LR schedule
    # -----------------------------------------------------------
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    total_steps = (len(loader) * epochs) // grad_accum

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return max(0.1, 1 - (step - warmup_steps) / (total_steps - warmup_steps))

    scheduler = LambdaLR(optimizer, lr_lambda)

    scaler = GradScaler()

    print("\n========== STARTING SFT TRAINING ==========\n")

    global_step = 0
    running_loss = 0.0

    # -----------------------------------------------------------
    # Training Loop
    # -----------------------------------------------------------
    for epoch in range(epochs):
        print(f"\n======= EPOCH {epoch+1}/{epochs} =======")

        for step, (x, labels) in enumerate(loader):
            x, labels = x.to(device), labels.to(device)

            # AMP autocast
            with autocast("cuda"):
                logits, loss = model(x, targets=labels)
                loss = loss / grad_accum

            scaler.scale(loss).backward()
            running_loss += loss.item()

            if (step + 1) % grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

                if global_step % 10 == 0:
                    print(f"[Step {global_step}] Loss: {running_loss/10:.4f}")
                    running_loss = 0.0

        # Save checkpoint
        ckpt_path = os.path.join(SAVE_DIR, f"sft_epoch{epoch+1}.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"[INFO] Saved → {ckpt_path}")

    print("\n========== SFT TRAINING COMPLETE ==========\n")


# ===============================================================
if __name__ == "__main__":
    train_sft()
