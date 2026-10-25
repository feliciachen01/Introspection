"""Run configuration, loaded from configs/*.yaml."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml


@dataclass
class Config:
    run_name: str
    models: list[str]
    layer_fractions: list[float]
    strengths: list[float]
    out_dir: str = "runs"
    device: str = "auto"
    dtype: str = "auto"
    load_in_4bit: bool = False          # applies to models of 7B and up
    batch_size: int = 25                # trials per generate() call
    max_new_tokens: int = 100           # paper: 100 tokens at temperature 1
    temperature: float = 1.0
    n_controls: int = 100               # no-injection trials per model (false positives)
    bias_strengths: list[float] = field(default_factory=lambda: [0.0, 2.0, 4.0])  # 0 = baseline yes-rate
    seed: int = 0
    judge_model: str = "Qwen/Qwen2.5-7B-Instruct"
    judge_backend: str = "hf"           # "vllm" on Colab (much faster), "hf" elsewhere
    judge_4bit: bool = True             # hf backend only
    judge_batch_size: int = 16
    judge_max_new_tokens: int = 192

    @property
    def run_dir(self) -> Path:
        return Path(self.out_dir) / self.run_name

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path: str | Path) -> Config:
    return Config(**yaml.safe_load(Path(path).read_text()))
