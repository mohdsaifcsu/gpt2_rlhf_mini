# GPT-2 Mini XL - Full RLHF Pipeline (SFT -> Reward Model -> PPO)

This repository implements a complete RLHF training pipeline on a compact GPT-2 Mini XL architecture.
It includes:

- Custom GPT-2 architecture (12 layers, 384-d, tied embeddings)
- Custom GPT-2 BPE tokenizer (GPT-2 style regex + merges)
- Supervised Fine-Tuning (SFT)
- Reward Modeling (RM)
- PPO Reinforcement Learning (RLHF)
- Chatbot inference with multi-turn dialogue
- Fully reproducible training scripts
- Clean folder structure + gitignore setup

This project is designed to serve as an end-to-end RLHF research template, inspired by GPT-2, PPO, and alignment literature - runnable on Colab or a single GPU.

---

## Project Structure
```bash
RLHF_GPT_MINI_XL/
│
├── checkpoints/                  
│   ├── pretrained/
│   ├── sft/
│   ├── reward_model/
│   └── ppo/
│
├── data/
│   ├── raw/
│   │   ├── pretrain_corpus.txt
│   │   └── prompts.txt
│   ├── sft/
│   │   └── sft_dataset.jsonl
│   ├── preferences/
│   │   └── pairs.jsonl
│   └── vocab/
│       ├── vocab.json
│       └── merges.txt
│
├── src/
│   ├── models/
│   │   └── gpt2_mini_xl.py
│   ├── tokenizer/
│   │   ├── bpe_tokenizer.py
│   │   └── mini_tokenizer.py
│   ├── rlhf/
│   │   ├── reward_model.py
│   │   ├── ppo_trainer.py
│   │   └── utils.py (optional)
│   └── training/
│       ├── train_sft.py
│       ├── train_reward_model.py
│       └── train_ppo.py
│
├── chatbot_inference.py
├── requirements.txt
└── .gitignore
```
---

## Tokenizer (GPT-2 BPE)

This project includes a fully working GPT-2 BPE tokenizer with:

- GPT-2 regex tokenization
- BPE merge rules
- Special tokens
- Dialogue formatting
- BOS/EOS
- Multi-turn chat support
```bash
src/tokenizer/bpe_tokenizer.py
data/vocab/vocab.json
data/vocab/merges.txt
```
---

## Model: GPT-2 Mini XL (12-Layer)

- 12 transformer blocks
- 384 embedding dimension
- 12 attention heads
- Weight tying
- Causal attention masks
- GPT-2 position embeddings
```bash
src/models/gpt2_mini_xl.py
```
---

## Stage 1 - Supervised Fine-Tuning (SFT)
#### Script:
```bash
python -m src.training.train_sft
```
#### Input:
```bash
data/sft/sft_dataset.jsonl
```
Each line is:
```bash
{"prompt": "......", "response": "...."}
```
#### Output:
Saved to:
```bash
checkpoints/sft/sft_epochX.pth
```
---

## Stage 2 - Reward Model Training (RM)
#### Script:
```bash
python -m src.training.train_reward_model
```
#### Input:
```bash
data/preferences/pairs.jsonl
```
Format:
```bash
{"prompt": ".......", "chosen": ".......", "rejected": "......."}
```
Trains a sequence-level reward model (scalar output).
#### Output:
```bash
checkpoints/reward_model/rm_epoch1.pth
```
---

## Stage 3 - PPO RLHF Training
#### Script:
```bash
python -m src.training.train_ppo
```
Uses:
- Actor (trainable)
- Reference (frozen)
- Reward model (frozen)
- GPT-2 BPE tokenizer
#### Input prompts:
```bash
data/raw/prompts.txt
```
#### Output:
```bash
checkpoints/ppo/actor_epoch1.pth
```
---

## Chatbot Inference
Run:
```bash
python chatbot_inference.py
```
You get an interactive chat:
```bash
You: Hello!
Assistant: Hi! How can I help you today?
```
Uses:
- RLHF PPO-fine-tuned model
- Multi-turn dialogue memory
- GPT-2 BPE tokenizer

---

## Installing Requirements
Install dependencies:
```bash
pip install -r requirements.txt
```
The requirements file includes:
```bash
torch
tqdm
regex
numpy
```
(You can add more depending on your environment.)

---

## GPU Requirements

- Colab T4 / L4 / A100
- Single RTX 3060/3070/3080/4090
- Works on 6-12 GB VRAM if batch sizes are small

---


## Goals of This Project

- Build a Fully Understandable RLHF Stack (from scratch)
- Create a Lightweight GPT-2 Model for Experimentation
- Provide a Modular, Clean Codebase
- Support Multi-Turn Chat and Real-World Use

---

## Future Improvements

- Add a Trainer for Pretraining / Continued Pretraining
- Add Weighted Losses for SFT (Stability Improvement)
- Add Better Tokenizer: GPT-2 BPE -> GPT-NeoX Tokenizer Upgrade
- Add KL-Adaptive PPO (Improved Alignment)
- Add OpenAI-Style System Prompt Formatting
- Add Evaluation Benchmarks
- Improve Reward Model Architecture
- Add LoRA Support for SFT and RLHF
- Add Quantization
- Add a Web Chat UI

---

##  Author

**Mohd Saif**  
Master’s Student - Colorado State University  
GitHub: https://github.com/mohdsaifcsu

---

##  License

This project is for **academic and educational use** only.

---
























