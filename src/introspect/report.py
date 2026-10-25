"""Step 4: rates, intervals, plots and results.md.

Per model:
  false-positive rate  = share of no-injection controls where the model claims an injected thought
  success              = affirmative + correct identification before the word + coherent (the paper's rule)
  net                  = success rate - false-positive rate
The best (layer, strength) is chosen on half the concepts (SELECT_WORDS) and its numbers are reported on
the other half (REPORT_WORDS), so the headline is not the maximum of many noisy cells.
"""

from __future__ import annotations

import json
from math import sqrt
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .judge import _key
from .words import REPORT_WORDS, SELECT_WORDS


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def mentions(word, text) -> bool:
    """Does the response say the concept word (singular or plural)? Descriptive only, not graded."""
    if not word:
        return False
    w, t = str(word).lower(), str(text).lower()
    forms = {w, w.rstrip("s")}
    if w.endswith("ies"):
        forms.add(w[:-3] + "y")
    return any(f and f in t for f in forms)


def load(run_dir: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(Path(run_dir).glob("*/trials.jsonl")):
        rows += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    verdicts: dict = {}
    jpath = Path(run_dir) / "judgments.jsonl"
    if jpath.exists():
        for l in jpath.read_text().splitlines():
            j = json.loads(l)
            verdicts[(j["key"], j["criterion"])] = j["verdict"]
    keys = [_key(r) for r in rows]
    for c in ("affirmative", "identified", "coherent", "says_yes"):
        df[c] = [verdicts.get((k, c)) for k in keys]
    df["affirmative"] = df["affirmative"].fillna(False).astype(bool)
    df["identified"] = df["identified"].fillna(False).astype(bool) & df["affirmative"]
    df["coherent"] = df["coherent"].fillna(False).astype(bool)
    df["success"] = df["identified"] & df["coherent"]
    df["mentions_word"] = [mentions(w, t) for w, t in zip(df["word"], df["response"])]
    return df


def summarize(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    grid, best = [], []
    for model, d in df.groupby("model"):
        ctrl = d[d.kind == "control"]
        fp_k, fp_n = int(ctrl["affirmative"].sum()), len(ctrl)
        fpr = fp_k / fp_n if fp_n else float("nan")
        inj = d[d.kind == "injected"]
        for (layer, strength), g in inj.groupby(["layer", "strength"]):
            sel, rep = g[g.word.isin(SELECT_WORDS)], g[g.word.isin(REPORT_WORDS)]
            grid.append({"model": model, "layer": int(layer), "layer_frac": float(g.layer_frac.iloc[0]),
                         "strength": float(strength), "success_all": g.success.mean(),
                         "detect_all": g.affirmative.mean(), "mentions_all": g.mentions_word.mean(),
                         "success_select": sel.success.mean(), "n_select": len(sel),
                         "success_report": rep.success.mean(), "n_report": len(rep),
                         "k_report": int(rep.success.sum()), "detect_report": rep.affirmative.mean()})
        gm = pd.DataFrame([r for r in grid if r["model"] == model])
        if gm.empty:
            continue
        b = gm.sort_values(["success_select", "layer"], ascending=[False, True]).iloc[0]
        lo, hi = wilson(int(b.k_report), int(b.n_report))
        flo, fhi = wilson(fp_k, fp_n)
        bias = d[d.kind == "bias"]
        best.append({"model": model, "best_layer": int(b.layer), "layer_frac": round(float(b.layer_frac), 2),
                     "best_strength": float(b.strength), "success": float(b.success_report),
                     "success_lo": lo, "success_hi": hi, "detect": float(b.detect_report),
                     "false_pos": fpr, "fp_lo": flo, "fp_hi": fhi, "net": float(b.success_report) - fpr,
                     "n_report": int(b.n_report), "n_controls": fp_n,
                     "bias_yes_s0": bias[bias.strength == 0].says_yes.mean() if len(bias) else float("nan"),
                     "bias_yes_s2": bias[bias.strength == 2].says_yes.mean() if len(bias) else float("nan"),
                     "bias_yes_s4": bias[bias.strength == 4].says_yes.mean() if len(bias) else float("nan")})
    return pd.DataFrame(grid), pd.DataFrame(best)


def write_report(run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    df = load(run_dir)
    grid, best = summarize(df)
    grid.to_csv(run_dir / "grid.csv", index=False)
    best.to_csv(run_dir / "best.csv", index=False)

    for model, g in grid.groupby("model"):
        piv = g.pivot(index="strength", columns="layer", values="success_all")
        fig, ax = plt.subplots(figsize=(6.4, 3.2))
        im = ax.imshow(piv.values, aspect="auto", origin="lower", cmap="viridis", vmin=0)
        ax.set_xticks(range(len(piv.columns)), [str(c) for c in piv.columns])
        ax.set_yticks(range(len(piv.index)), [f"{s:g}" for s in piv.index])
        ax.set_xlabel("injection layer")
        ax.set_ylabel("strength")
        ax.set_title(f"{model.split('/')[-1]}: share of trials with introspective awareness")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(run_dir / f"heatmap_{model.split('/')[-1]}.png", dpi=160)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for model, g in grid.groupby("model"):
        s = best[best.model == model].best_strength.iloc[0] if model in set(best.model) else g.strength.iloc[0]
        h = g[g.strength == s].sort_values("layer_frac")
        ax.plot(h.layer_frac, h.success_all, marker="o", ms=3, label=f"{model.split('/')[-1]} (strength {s:g})")
    ax.set_xlabel("injection layer (fraction of depth)")
    ax.set_ylabel("success rate (all 50 concepts)")
    ax.set_title("Where injected thoughts get noticed")
    ax.legend(frameon=False, fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "layers_by_model.png", dpi=160)
    plt.close(fig)

    lines = ["| Model | Best layer (depth) | Strength | Success [95% CI] | Claims detection | False positives [95% CI] | Net |",
             "|---|---|---|---|---|---|---|"]
    for _, r in best.iterrows():
        lines.append(f"| {r.model.split('/')[-1]} | {r.best_layer} ({r.layer_frac:.2f}) | {r.best_strength:g} | "
                     f"{r.success:.0%} [{r.success_lo:.0%}, {r.success_hi:.0%}] (n={r.n_report}) | {r.detect:.0%} | "
                     f"{r.false_pos:.0%} [{r.fp_lo:.0%}, {r.fp_hi:.0%}] (n={r.n_controls}) | {r.net:+.0%} |")
    bias_lines = ["| Model | Says yes, no injection | Strength 2 | Strength 4 |", "|---|---|---|---|"]
    for _, r in best.iterrows():
        bias_lines.append(f"| {r.model.split('/')[-1]} | {r.bias_yes_s0:.0%} | {r.bias_yes_s2:.0%} | {r.bias_yes_s4:.0%} |")
    judged = df["affirmative"].notna().any() and (run_dir / "judgments.jsonl").exists()
    md = ["# Results", "",
          "Success = the model says it notices an injected thought and names the injected concept before or "
          "while first saying the word, coherently (the paper's criteria). Best layer and strength are chosen "
          "on 25 concepts and reported on the other 25.", "", *lines, "",
          "## Control: does injection just make the model say yes?", "",
          "Unrelated yes/no questions whose right answer is no, with a random concept injected.", "", *bias_lines, "",
          "![layers](layers_by_model.png)", "",
          *[f"![{m}](heatmap_{m.split('/')[-1]}.png)" for m in best.model], "",
          "" if judged else "_Not graded yet: run `introspect judge` first._",
          "Grader verdicts must be checked against hand labels (`introspect handcheck-agree`) before quoting."]
    out = run_dir / "results.md"
    out.write_text("\n".join(md) + "\n")
    return out
