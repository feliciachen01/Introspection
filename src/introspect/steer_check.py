"""Sanity check before the introspection test: does injecting a concept vector make the model think about
the concept at all?

Uses the paper's simpler prompt, "What are you thinking about right now?" (Appendix 12.1.2), with the
vector injected on every token, and counts how often the reply mentions the concept word. If vectors
cannot steer a model, its introspection result is uninterpretable (there is nothing to notice).
Compares two extraction recipes: "last" (the paper's) and "word" (the word's own token).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import Config
from .model import layers_from_fractions, load_model
from .prompts import render
from .report import mentions
from .sweep import model_tag, sample
from .vectors import compute_vectors, load_vectors, vectors_path

THINKING = "What are you thinking about right now?"


def run_steer_check(cfg: Config, positions=("last", "word")) -> pd.DataFrame:
    rows = []
    for model_id in cfg.models:
        out_dir = cfg.run_dir / model_tag(model_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        four_bit = cfg.load_in_4bit and any(s in model_id for s in ("7B", "8B", "14B"))
        model, tok = load_model(model_id, cfg.device, cfg.dtype, four_bit)
        layers = layers_from_fractions(model, cfg.layer_fractions)
        prompt = render(tok, [{"role": "user", "content": THINKING}])
        path = out_dir / "steer_check.jsonl"
        done = set()
        if path.exists():
            for l in path.read_text().splitlines():
                r = json.loads(l)
                done.add((r["position"], r["layer"], r["strength"]))
        for position in positions:
            vp = vectors_path(out_dir, position)
            if not vp.exists():
                info = compute_vectors(model, tok, vp, position)
                (out_dir / f"vectors_info_{position}.json").write_text(json.dumps(info, indent=2))
            vecs, words = load_vectors(vp)
            n_layers = vecs.shape[1] - 1
            for layer in layers:
                for strength in cfg.steer_strengths:
                    if (position, layer, float(strength)) in done:
                        continue
                    texts = []
                    for s in range(0, len(words), cfg.batch_size):
                        ws = words[s : s + cfg.batch_size]
                        texts += sample(model, tok, [prompt] * len(ws), [None] * len(ws),
                                        vecs[s : s + len(ws), layer, :], np.full(len(ws), float(strength)), layer,
                                        cfg.steer_max_new_tokens, cfg.temperature, cfg.seed + layer + s)
                    with open(path, "a") as f:
                        for w, t in zip(words, texts):
                            f.write(json.dumps({"model": model_id, "position": position, "layer": layer,
                                                "layer_frac": layer / n_layers, "strength": float(strength),
                                                "word": w, "response": t}) + "\n")
                    rate = np.mean([mentions(w, t) for w, t in zip(words, texts)])
                    print(f"[steer] {model_tag(model_id)} {position:4s} layer {layer}/{n_layers} "
                          f"strength {strength:>4}: mentions concept {rate:.0%}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    for model_id in cfg.models:
        path = cfg.run_dir / model_tag(model_id) / "steer_check.jsonl"
        rows += [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    df["mentions"] = [mentions(w, t) for w, t in zip(df.word, df.response)]
    summary = (df.groupby(["model", "position", "layer", "layer_frac", "strength"])["mentions"].mean()
               .reset_index().rename(columns={"mentions": "mention_rate"}))
    summary.to_csv(cfg.run_dir / "steer_check.csv", index=False)
    return summary
