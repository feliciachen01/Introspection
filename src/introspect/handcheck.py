"""Check the open grader against your own judgement.

export: 200 responses to label, oversampling the cases that matter (grader successes, every "yes" answer
        that also says the concept word, which is where the grader can miss a success, other "yes" answers,
        and control trials), with the grader's verdicts in the last columns, not to be looked at while labeling.
agree:  agreement and Cohen's kappa for "claims detection" and "success".
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from .report import load


def export(run_dir: Path, n: int = 200, seed: int = 0) -> Path:
    df = load(run_dir)
    df = df[df.kind.isin(["injected", "control"])]
    near = df.affirmative & df.mentions_word & ~df.success
    parts = [df[df.success].sample(n=min(60, int(df.success.sum())), random_state=seed),
             df[near].sample(n=min(40, int(near.sum())), random_state=seed)]
    rest_yes = df[df.affirmative & ~df.success & ~near]
    parts.append(rest_yes.sample(n=min(60, len(rest_yes)), random_state=seed))
    ctrl = df[(df.kind == "control") & ~df.affirmative]
    parts.append(ctrl.sample(n=min(30, len(ctrl)), random_state=seed))
    taken = pd.concat(parts)
    rest = df.drop(taken.index)
    parts.append(rest.sample(n=min(max(0, n - len(taken)), len(rest)), random_state=seed))
    out = pd.concat(parts).sample(frac=1, random_state=seed)
    out = out[["model", "kind", "layer", "strength", "word", "response", "affirmative", "success"]].rename(
        columns={"affirmative": "grader_affirmative", "success": "grader_success"})
    out["human_affirmative"] = ""
    out["human_success"] = ""
    cols = ["model", "kind", "layer", "strength", "word", "response", "human_affirmative", "human_success",
            "grader_affirmative", "grader_success"]
    path = Path(run_dir) / "handcheck.csv"
    out[cols].to_csv(path, index=False)
    return path


def agree(run_dir: Path) -> dict:
    df = pd.read_csv(Path(run_dir) / "handcheck.csv")
    res = {}
    for human, grader in (("human_affirmative", "grader_affirmative"), ("human_success", "grader_success")):
        d = df[df[human].astype(str).str.strip().isin(["0", "1", "0.0", "1.0"])]
        if d.empty:
            continue
        h, g = d[human].astype(float).astype(int), d[grader].astype(bool).astype(int)
        res[human.replace("human_", "")] = {"n": int(len(d)), "agreement": float((h == g).mean()),
                                            "kappa": float(cohen_kappa_score(h, g)),
                                            "grader_yes_human_no": int(((g == 1) & (h == 0)).sum()),
                                            "grader_no_human_yes": int(((g == 0) & (h == 1)).sum())}
    return res
