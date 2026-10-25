"""Command line: introspect <step> --config configs/<file>.yaml"""

from __future__ import annotations

import argparse
import json

from .config import load_config


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="introspect", description="Can it notice its own thoughts?")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, help_ in [("sweep", "concept vectors + injected, control and bias trials for every model"),
                        ("judge", "grade responses with the open grader"),
                        ("report", "results.md, grid.csv, best.csv and plots"),
                        ("all", "sweep, judge, report"),
                        ("steer-check", "does injection make the model think about the concept at all?"),
                        ("handcheck-export", "write 200 responses to label by hand"),
                        ("handcheck-agree", "compare hand labels with the grader")]:
        sub.add_parser(name, help=help_).add_argument("--config", required=True)
    a = p.parse_args(argv)
    cfg = load_config(a.config)
    if a.cmd in ("sweep", "all"):
        from .sweep import run_sweep

        run_sweep(cfg)
    if a.cmd in ("judge", "all"):
        from .judge import judge_run

        judge_run(cfg)
    if a.cmd in ("report", "all"):
        from .report import write_report

        print(f"wrote {write_report(cfg.run_dir)}")
    if a.cmd == "steer-check":
        from .steer_check import run_steer_check

        s = run_steer_check(cfg)
        best = s.sort_values("mention_rate", ascending=False).groupby(["model", "position"]).head(1)
        print(best.to_string(index=False))
    if a.cmd == "handcheck-export":
        from .handcheck import export

        print(f"wrote {export(cfg.run_dir)}")
    if a.cmd == "handcheck-agree":
        from .handcheck import agree

        print(json.dumps(agree(cfg.run_dir), indent=2))


if __name__ == "__main__":
    main()
