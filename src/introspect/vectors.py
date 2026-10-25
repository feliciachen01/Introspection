"""Concept vectors (Lindsey et al. 2026, Sec. 5.2 and Appendix 12.1.1).

vector(word, layer) = activation at the last prompt token of "Tell me about {word}." minus the mean of
the same activation over 100 baseline words. Computed for every layer, so the injection layer can be
swept with the vector taken from that same layer, as in the paper.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .prompts import concept_messages, render
from .words import BASELINE_WORDS, CONCEPT_WORDS


@torch.no_grad()
def last_token_states(model, tok, words: list[str], batch_size: int = 16) -> np.ndarray:
    """(n_words, L+1, D) hidden states at the last prompt token, all layers."""
    out = []
    for i in range(0, len(words), batch_size):
        prompts = [render(tok, concept_messages(w)) for w in words[i : i + batch_size]]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        hs = model(**enc, output_hidden_states=True).hidden_states
        out.append(torch.stack([h[:, -1, :] for h in hs], dim=1).float().cpu().numpy())
    return np.concatenate(out)


def compute_vectors(model, tok, out_path: Path) -> dict:
    base = last_token_states(model, tok, BASELINE_WORDS)
    conc = last_token_states(model, tok, CONCEPT_WORDS)
    mean = base.mean(axis=0)                                  # (L+1, D)
    vecs = conc - mean[None]                                  # (50, L+1, D)
    resid_norm = np.linalg.norm(base, axis=-1).mean(axis=0)   # (L+1,) typical residual size per layer
    vec_norm = np.linalg.norm(vecs, axis=-1).mean(axis=0)     # (L+1,) typical concept-vector size
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, vectors=vecs.astype(np.float32), words=np.array(CONCEPT_WORDS),
             resid_norm=resid_norm, vec_norm=vec_norm)
    return {"n_layers": int(vecs.shape[1] - 1), "dim": int(vecs.shape[2]),
            "vec_to_resid_ratio": [round(float(a / b), 3) for a, b in zip(vec_norm, resid_norm)]}


def load_vectors(path: Path) -> tuple[np.ndarray, list[str]]:
    z = np.load(path)
    return z["vectors"], z["words"].tolist()
