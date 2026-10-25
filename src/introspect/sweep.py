"""Step 2: generate responses under concept injection.

For each model, writes runs/<run>/<model>/trials.jsonl with one row per sampled response:
  injected  50 concepts x every (layer, strength): the "Trial 1" prompt with the concept injected
  control   n_controls trials of the same prompt with no injection (false positives)
  bias      the 20 unrelated yes/no questions with a random concept injected (does injection make
            the model say "yes" to anything?)
Sampling follows the paper: temperature 1, 100 new tokens. Re-running skips finished configurations.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import numpy as np
import torch

from .config import Config
from .inject import inject
from .model import layers_from_fractions, load_model
from .prompts import (BIAS_QUESTIONS, INJECT_FROM, bias_messages, inject_start_token, render, trial_messages)
from .vectors import compute_vectors, load_vectors


def model_tag(model_id: str) -> str:
    """Folder name for a model: the last part of a Hub ID or a local path (Windows or POSIX)."""
    return re.split(r"[\\/]", model_id.rstrip("/\\"))[-1]


@torch.no_grad()
def sample(model, tok, prompts: list[str], markers: list[str | None], vectors: np.ndarray, strengths: np.ndarray,
           layer: int | None, max_new_tokens: int, temperature: float, seed: int) -> list[str]:
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    T = enc["input_ids"].shape[1]
    lengths = enc["attention_mask"].sum(dim=1).tolist()
    starts = np.array([(T - n) + inject_start_token(tok, p, m) for p, m, n in zip(prompts, markers, lengths)])
    enc = enc.to(model.device)
    torch.manual_seed(seed)
    # pure temperature sampling, overriding the model's default top_p / top_k / repetition penalty
    gen = dict(max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature, top_p=1.0, top_k=0,
               repetition_penalty=1.0, pad_token_id=tok.pad_token_id)
    if layer is None or not np.any(strengths):
        out = model.generate(**enc, **gen)
    else:
        with inject(model, layer, vectors, strengths, starts):
            out = model.generate(**enc, **gen)
    return tok.batch_decode(out[:, T:], skip_special_tokens=True)


def _done(path: Path) -> set:
    if not path.exists():
        return set()
    keys = set()
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            keys.add((r["kind"], r["layer"], r["strength"], r["item"]))
    return keys


def run_model(cfg: Config, model_id: str) -> Path:
    out_dir = cfg.run_dir / model_tag(model_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    four_bit = cfg.load_in_4bit and any(s in model_id for s in ("7B", "8B", "14B"))
    model, tok = load_model(model_id, cfg.device, cfg.dtype, four_bit)
    vec_path = out_dir / "vectors.npz"
    if not vec_path.exists():
        info = compute_vectors(model, tok, vec_path)
        (out_dir / "vectors_info.json").write_text(json.dumps(info, indent=2))
        print(f"[vectors] {model_id}: {info['n_layers']} layers, dim {info['dim']}")
    vecs, words = load_vectors(vec_path)
    layers = layers_from_fractions(model, cfg.layer_fractions)
    n_layers = vecs.shape[1] - 1
    path = out_dir / "trials.jsonl"
    done = _done(path)
    trial_prompt = render(tok, trial_messages())
    D = vecs.shape[2]

    def write(rows):
        with open(path, "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    # controls: no injection
    todo = [i for i in range(cfg.n_controls) if ("control", None, 0.0, str(i)) not in done]
    for s in range(0, len(todo), cfg.batch_size):
        idx = todo[s : s + cfg.batch_size]
        texts = sample(model, tok, [trial_prompt] * len(idx), [INJECT_FROM] * len(idx), np.zeros((len(idx), D)),
                       np.zeros(len(idx)), None, cfg.max_new_tokens, cfg.temperature, cfg.seed + 7919 + s)
        write([{"model": model_id, "kind": "control", "layer": None, "layer_frac": None, "strength": 0.0,
                "item": str(i), "word": None, "response": t} for i, t in zip(idx, texts)])
    print(f"[sweep] {model_id}: controls done")

    for layer in layers:
        for strength in cfg.strengths:
            todo = [w for w in words if ("injected", layer, float(strength), w) not in done]
            for s in range(0, len(todo), cfg.batch_size):
                ws = todo[s : s + cfg.batch_size]
                v = vecs[[words.index(w) for w in ws], layer, :]
                texts = sample(model, tok, [trial_prompt] * len(ws), [INJECT_FROM] * len(ws), v,
                               np.full(len(ws), float(strength)), layer, cfg.max_new_tokens, cfg.temperature,
                               cfg.seed + layer * 1000 + int(strength * 10) + s)
                write([{"model": model_id, "kind": "injected", "layer": layer, "layer_frac": layer / n_layers,
                        "strength": float(strength), "item": w, "word": w, "response": t} for w, t in zip(ws, texts)])
            print(f"[sweep] {model_id}: layer {layer}/{n_layers}, strength {strength}")
        for strength in cfg.bias_strengths:
            todo = [q for q in BIAS_QUESTIONS if ("bias", layer, float(strength), q) not in done]
            if not todo:
                continue
            rng = random.Random(f"{cfg.seed}-{layer}-{strength}")
            ws = [rng.choice(words) for _ in todo]
            prompts = [render(tok, bias_messages(q)) for q in todo]
            v = vecs[[words.index(w) for w in ws], layer, :]
            texts = sample(model, tok, prompts, todo, v, np.full(len(todo), float(strength)),
                           layer if strength else None, cfg.max_new_tokens, cfg.temperature,
                           cfg.seed + 31 * layer + int(strength))
            write([{"model": model_id, "kind": "bias", "layer": layer, "layer_frac": layer / n_layers,
                    "strength": float(strength), "item": q, "word": w, "question": q, "response": t}
                   for q, w, t in zip(todo, ws, texts)])
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return path


def run_sweep(cfg: Config) -> None:
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    (cfg.run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    for m in cfg.models:
        run_model(cfg, m)
