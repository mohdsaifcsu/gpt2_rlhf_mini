# ================================================================
# chatbot_inference.py - CHAT INTERFACE FOR PPO TRAINED MODEL
# Compatible with:
#  - GPT2BPETokenizer
#  - GPT2MiniXL
#  - PPO-trained actor (actor_epochX.pth)
# ================================================================

import os
import torch

from src.tokenizer.bpe_tokenizer import GPT2BPETokenizer
from src.models.gpt2_mini_xl import GPT2MiniXL


# ================================================================
# Load Model
# ================================================================
def load_model(checkpoint_path, tokenizer, device):
    print(f"[INFO] Loading model -> {checkpoint_path}")

    model = GPT2MiniXL(
        vocab_size=tokenizer.vocab_size,
        block_size=256,
        embed_dim=384,
        num_layers=12,
        num_heads=12
    ).to(device)

    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state, strict=False)

    model.eval()
    return model


# ================================================================
# Generate assistant response
# ================================================================
@torch.no_grad()
def generate_response(model, tokenizer, dialogue, max_new_tokens=150, top_k=50):

    """
    dialogue = [
        ("system", "..."),
        ("user", "..."),
        ("assistant", "..."),
        ...
    ]
    """

    # Convert to token IDs (NO bos/eos inside each turn)
    input_ids = tokenizer.format_dialogue(dialogue)

    x = torch.tensor([input_ids], dtype=torch.long, device=model.wte.weight.device)

    # Generate continuation
    out = model.generate(
        idx=x,
        max_new_tokens=max_new_tokens,
        temperature=1.0,
        top_k=top_k
    )

    out = out[0].tolist()

    new_tokens = out[len(input_ids):]

    reply = tokenizer.decode(new_tokens).strip()
    return reply


# ================================================================
# Interactive Chat Loop
# ================================================================
def chat_loop(model, tokenizer, device):

    print("\n==============================")
    print("        RLHF Chatbot")
    print("==============================\n")
    print("Type 'quit' to exit.\n")

    # 1st message: system instruction
    history = [
        ("system", "You are a helpful, friendly AI assistant.")
    ]

    while True:
        user = input("User: ").strip()
        if user.lower() in ("quit", "exit"):
            break

        history.append(("user", user))

        reply = generate_response(model, tokenizer, history)
        print("\nAssistant:", reply, "\n")

        history.append(("assistant", reply))


# ================================================================
# Main
# ================================================================
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------
    # Load tokenizer
    # ------------------------------------------------------------
    tokenizer = GPT2BPETokenizer(
        vocab_file="data/vocab/vocab.json",
        merges_file="data/vocab/merges.txt"
    )

    # ------------------------------------------------------------
    # Choose checkpoint
    # ------------------------------------------------------------
    ckpt = "checkpoints/ppo/actor_epoch1.pth"

    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint does not exist -> {ckpt}")

    # ------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------
    model = load_model(ckpt, tokenizer, device)

    # ------------------------------------------------------------
    # Start chat
    # ------------------------------------------------------------
    chat_loop(model, tokenizer, device)
