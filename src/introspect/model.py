"""Model loading and access to the residual stream."""

from __future__ import annotations

import torch


def pick_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(model_id: str, device: str = "auto", dtype: str = "auto", load_in_4bit: bool = False):
    import transformers
    from packaging.version import Version
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = pick_device(device)
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    torch_dtype = {"auto": torch.bfloat16 if device == "cuda" else torch.float32, "float32": torch.float32,
                   "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    key = "dtype" if Version(transformers.__version__) >= Version("4.56") else "torch_dtype"
    kwargs = {key: torch_dtype}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs = {"quantization_config": BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16),
                  "device_map": "auto"}
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if not load_in_4bit:
        model.to(device)
    model.eval()
    return model, tok


def decoder_layers(model):
    for path in ("model.layers", "transformer.h", "gpt_neox.layers"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    raise ValueError("unsupported architecture: cannot find decoder layers")


def num_layers(model) -> int:
    return len(decoder_layers(model))


def layers_from_fractions(model, fractions: list[float]) -> list[int]:
    """Residual-stream positions to inject at, as hidden_states indices 1..L-1 (index l = output of
    decoder layer l-1). The last index is excluded because HF returns it after the final norm."""
    n = num_layers(model)
    out = sorted({min(max(1, round(f * n)), n - 1) for f in fractions})
    return out
