"""A compact byte-pair tokenizer implemented for this project."""
from collections import Counter
import json
from pathlib import Path

class BPETokenizer:
    def __init__(self):
        self.token_to_id = {}
        self.id_to_token = {}
        self.merges = []

    @property
    def vocab_size(self):
        return len(self.token_to_id)

    def train(self, texts, vocab_size=500, min_frequency=2):
        words = Counter()
        for text in texts:
            for word in text.strip().split():
                if word:
                    words[tuple(list(word) + ["</w>"])] += 1
        if not words:
            raise ValueError("Cannot train tokenizer on an empty corpus.")
        vocab = {word: freq for word, freq in words.items()}
        alphabet = sorted({ch for word in vocab for ch in word})
        tokens = alphabet + (["</w>"] if "</w>" not in alphabet else [])
        self.merges = []
        while len(tokens) < vocab_size:
            pairs = Counter()
            for word, freq in vocab.items():
                for i in range(len(word) - 1):
                    pairs[(word[i], word[i + 1])] += freq
            if not pairs:
                break
            pair, count = pairs.most_common(1)[0]
            if count < min_frequency:
                break
            merged = pair[0] + pair[1]
            self.merges.append(pair)
            new_vocab = {}
            for word, freq in vocab.items():
                out, i = [], 0
                while i < len(word):
                    if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                        out.append(merged)
                        i += 2
                    else:
                        out.append(word[i])
                        i += 1
                new_vocab[tuple(out)] = new_vocab.get(tuple(out), 0) + freq
            vocab = new_vocab
            if merged not in tokens:
                tokens.append(merged)
        specials = ["<pad>", "<unk>", "<bos>", "<eos>"]
        ordered = specials + [t for t in tokens if t not in specials]
        self.token_to_id = {t: i for i, t in enumerate(ordered)}
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}
        return self

    def encode(self, text, add_special_tokens=False):
        ids = []
        for word in text.strip().split():
            pieces = list(word) + ["</w>"]
            for left, right in self.merges:
                merged, i = [], 0
                while i < len(pieces):
                    if i + 1 < len(pieces) and pieces[i] == left and pieces[i + 1] == right:
                        merged.append(left + right)
                        i += 2
                    else:
                        merged.append(pieces[i])
                        i += 1
                pieces = merged
            ids.extend(self.token_to_id.get(piece, self.token_to_id["<unk>"]) for piece in pieces)
        if add_special_tokens:
            ids = [self.token_to_id["<bos>"]] + ids + [self.token_to_id["<eos>"]]
        return ids

    def decode(self, ids, skip_special_tokens=True):
        specials = {"<pad>", "<unk>", "<bos>", "<eos>"}
        text = "".join(
            self.id_to_token.get(int(i), "<unk>")
            for i in ids
            if not (skip_special_tokens and self.id_to_token.get(int(i)) in specials)
        )
        return text.replace("</w>", " ").strip()

    def save(self, path):
        Path(path).write_text(json.dumps({
            "token_to_id": self.token_to_id,
            "merges": self.merges
        }, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        obj = cls()
        obj.token_to_id = {k: int(v) for k, v in raw["token_to_id"].items()}
        obj.id_to_token = {v: k for k, v in obj.token_to_id.items()}
        obj.merges = [tuple(x) for x in raw["merges"]]
        return obj
