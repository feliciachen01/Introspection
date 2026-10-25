"""Concept vectors (Lindsey et al. 2026, Sec. 5.2 and Appendix 12.1.1).

vector(word, layer) = activation in "Tell me about {word}." minus the mean of the same activation over
100 baseline words, at every layer, so the injection layer can be swept with the vector taken from that
same layer, as in the paper.

position="last" (the paper): the last prompt token, just before the reply.
position="word": the last token of the word itself. Added because the 0.5B pilot showed "last" vectors
barely steer a small model; this checks whether the recipe, rather than the model, is the problem.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .prompts import concept_messages, render
from .words import BASELINE_WORDS, CONCEPT_WORDS


def _word_token_index(tok, text: str, word: str) -> int:
    """Index (unpadded) of the token covering the last character of `word` in "... about {word}."."""
    start = text.rfind(f"about {word}") + len("about ")
    last_char = start + len(word) - 1
    offsets = tok(text, return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]
    for i, (s, e) in enumerate(offsets):
        if s <= last_char < e:
            return i
    raise ValueError(f"word {word!r} not found in tokenized prompt")


@torch.no_grad()
def token_states(model, tok, words: list[str], position: str = "last", batch_size: int = 16) -> np.ndarray:
    """(n_words, L+1, D) hidden states at the chosen position, all layers."""
    out = []
    for i in range(0, len(words), batch_size):
        chunk = words[i : i + batch_size]
        prompts = [render(tok, concept_messages(w)) for w in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
        T = enc["input_ids"].shape[1]
        if position == "last":
            idx = [T - 1] * len(chunk)
        else:
            lengths = enc["attention_mask"].sum(dim=1).tolist()
            idx = [(T - n) + _word_token_index(tok, p, w) for p, w, n in zip(prompts, chunk, lengths)]
        hs = model(**enc.to(model.device), output_hidden_states=True).hidden_states
        rows = torch.arange(len(chunk), device=hs[0].device)
        pos = torch.tensor(idx, device=hs[0].device)
        out.append(torch.stack([h[rows, pos, :] for h in hs], dim=1).float().cpu().numpy())
    return np.concatenate(out)


def compute_vectors(model, tok, out_path: Path, position: str = "last") -> dict:
    base = token_states(model, tok, BASELINE_WORDS, position)
    conc = token_states(model, tok, CONCEPT_WORDS, position)
    mean = base.mean(axis=0)                                  # (L+1, D)
    vecs = conc - mean[None]                                  # (50, L+1, D)
    resid_norm = np.linalg.norm(base, axis=-1).mean(axis=0)   # (L+1,) typical residual size per layer
    vec_norm = np.linalg.norm(vecs, axis=-1).mean(axis=0)     # (L+1,) typical concept-vector size
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, vectors=vecs.astype(np.float32), words=np.array(CONCEPT_WORDS),
             resid_norm=resid_norm, vec_norm=vec_norm)
    return {"position": position, "n_layers": int(vecs.shape[1] - 1), "dim": int(vecs.shape[2]),
            "vec_to_resid_ratio": [round(float(a / b), 3) for a, b in zip(vec_norm, resid_norm)]}


def vectors_path(out_dir: Path, position: str) -> Path:
    """The paper's recipe keeps the original file name, so finished runs stay valid."""
    return out_dir / ("vectors.npz" if position == "last" else f"vectors_{position}.npz")


def load_vectors(path: Path) -> tuple[np.ndarray, list[str]]:
    z = np.load(path)
    return z["vectors"], z["words"].tolist()
