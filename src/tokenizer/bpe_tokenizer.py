# ============================================================
# GPT-2 BPE Tokenizer (Stable RLHF Version)
# ============================================================

import json
import regex as re
from typing import List


class GPT2BPETokenizer:

    def __init__(self, vocab_file: str, merges_file: str):

        # ------------------------
        # Load vocabulary
        # ------------------------
        with open(vocab_file, "r", encoding="utf-8") as f:
            self.encoder = json.load(f)

        self.decoder = {v: k for k, v in self.encoder.items()}

        # ------------------------
        # Add essential special tokens
        # ------------------------
        special = [
            "<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>",
            "<|user|>", "<|assistant|>", "<|system|>"
        ]
        for tok in special:
            if tok not in self.encoder:
                self.encoder[tok] = len(self.encoder)
                self.decoder[self.encoder[tok]] = tok

        self.pad_id = self.encoder["<|pad|>"]
        self.bos_id = self.encoder["<|bos|>"]
        self.eos_id = self.encoder["<|eos|>"]
        self.unk_id = self.encoder["<|unk|>"]

        self.user_id = self.encoder["<|user|>"]
        self.assistant_id = self.encoder["<|assistant|>"]
        self.system_id = self.encoder["<|system|>"]

        self.vocab_size = len(self.encoder)

        # ------------------------
        # Load merges
        # ------------------------
        merges = []
        with open(merges_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.startswith("#"):
                    parts = line.strip().split()
                    if len(parts) == 2:
                        merges.append(tuple(parts))

        self.bpe_ranks = {pair: i for i, pair in enumerate(merges)}

        # GPT-2 regex for splitting text
        self.pat = re.compile(
            r"'s|'t|'re|'ve|'m|'ll|'d"
            r"| ?\p{L}+| ?\p{N}+"
            r"| ?[^\s\p{L}\p{N}]+" 
            r"|\s+(?!\S)|\s+"
        )

    # ============================================================
    # BPE MERGING
    # ============================================================
    def get_pairs(self, word):
        pairs = set()
        if len(word) < 2:
            return pairs
        prev = word[0]
        for ch in word[1:]:
            pairs.add((prev, ch))
            prev = ch
        return pairs

    def bpe(self, token: str):
        if token in self.encoder:
            return token

        word = list(token)
        pairs = self.get_pairs(word)
        if not pairs:
            return token

        while True:
            min_rank = 1e10
            min_pair = None

            for pair in pairs:
                if pair in self.bpe_ranks and self.bpe_ranks[pair] < min_rank:
                    min_rank = self.bpe_ranks[pair]
                    min_pair = pair

            if min_pair is None:
                break

            first, second = min_pair
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break

                new_word.extend(word[i:j])
                i = j
                if i < len(word)-1 and word[i] == first and word[i+1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            word = new_word
            pairs = self.get_pairs(word)

        return " ".join(word)

    # ============================================================
    # Encoding
    # ============================================================
    def encode(self, text: str, add_special_tokens=True) -> List[int]:
        ids = []

        if add_special_tokens:
            ids.append(self.bos_id)

        tokens = re.findall(self.pat, text)
        for tok in tokens:
            tok = tok.strip()
            if not tok:
                continue
            merged = self.bpe(tok)
            for bt in merged.split(" "):
                ids.append(self.encoder.get(bt, self.unk_id))

        if add_special_tokens:
            ids.append(self.eos_id)

        return ids

    # ============================================================
    # Dialogue formatting
    # ============================================================
    def encode_turn(self, role, text):
        role_ids = {
            "user": self.user_id,
            "assistant": self.assistant_id,
            "system": self.system_id,
        }
        ids = [role_ids[role]]
        ids.extend(self.encode(text, add_special_tokens=False))
        return ids

    def format_dialogue(self, turns):
        out = []
        for role, txt in turns:
            out.extend(self.encode_turn(role, txt))
        out.append(self.eos_id)
        return out

    # ============================================================
    # Decoding
    # ============================================================
    def decode(self, ids: List[int]):
        toks = []
        for i in ids:
            if i in (
                self.bos_id, self.eos_id, self.pad_id,
                self.user_id, self.assistant_id, self.system_id
            ):
                continue
            toks.append(self.decoder.get(i, "<|unk|>"))
        text = "".join(toks)
        return text.replace("Ġ", " ").strip()
