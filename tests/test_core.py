"""Offline tests on a tiny randomly initialised model. They check the plumbing and the injection math;
they say nothing about introspection (a random model has none)."""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from introspect.config import Config  # noqa: E402
from introspect.inject import inject  # noqa: E402
from introspect.judge import parse_verdict  # noqa: E402
from introspect.prompts import INJECT_FROM, inject_start_token, render, trial_messages  # noqa: E402
from introspect.report import mentions, wilson  # noqa: E402
from introspect.words import BASELINE_WORDS, CONCEPT_WORDS, REPORT_WORDS, SELECT_WORDS  # noqa: E402


def make_tiny(path):
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    words = ["<pad>", "<eos>", "<unk>", "trial", "1", ":", "human", "assistant", "ok", ".", ",", "tell", "me", "about",
             "do", "you", "detect", "an", "injected", "thought", "?", "yes", "no", "let", "'", "s", "get", "started",
             "if", "so", "what", "is", "the", "Trial", "Human", "Assistant", "Ok", "Tell", "Do", "If", "I", "Let",
             "What"] + CONCEPT_WORDS
    vocab = {w: i for i, w in enumerate(dict.fromkeys(words))}
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>")
    tok.save_pretrained(path)
    cfg = Qwen2Config(vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=6,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
                      eos_token_id=1, pad_token_id=0, bos_token_id=1, tie_word_embeddings=True)
    torch.manual_seed(0)
    Qwen2ForCausalLM(cfg).save_pretrained(path)


def _load_tiny(model_id, *args, **kwargs):
    """Newer transformers versions swap a toy tokenizer for the model family's own tokenizer class when
    loaded through AutoTokenizer, so the tests load it exactly as built. Real Qwen models load normally."""
    from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

    tok = PreTrainedTokenizerFast.from_pretrained(model_id)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_id).eval()
    return model, tok


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    path = tmp_path_factory.mktemp("tiny")
    make_tiny(path)
    import introspect.judge
    import introspect.model
    import introspect.steer_check
    import introspect.sweep

    for mod in (introspect.model, introspect.sweep, introspect.steer_check, introspect.judge):
        mod.load_model = _load_tiny
    model, tok = _load_tiny(str(path))
    return str(path), model, tok


def test_word_lists_match_the_paper():
    assert len(BASELINE_WORDS) == 100 and len(CONCEPT_WORDS) == 50
    assert CONCEPT_WORDS[0] == "dust" and CONCEPT_WORDS[-1] == "silver"
    assert set(SELECT_WORDS).isdisjoint(REPORT_WORDS) and len(SELECT_WORDS) == len(REPORT_WORDS) == 25


def test_injection_starts_at_trial_1(tiny):
    _, _, tok = tiny
    text = render(tok, trial_messages())
    i = inject_start_token(tok, text, INJECT_FROM)
    offsets = tok(text, return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]
    char = text.rfind(INJECT_FROM)
    # contract: every earlier token ends before the marker; the start token reaches into or past it
    assert all(end <= char for _, end in offsets[:i])
    assert offsets[i][1] > char
    # the marker is the last "\n\nTrial 1", in the final user turn, after the setup text
    assert char > text.rfind("control trials")
    # this toy tokenizer drops whitespace, so the start token is the word "Trial" itself
    ids = tok(text, add_special_tokens=False)["input_ids"]
    assert tok.convert_ids_to_tokens(ids)[i] == "Trial"


def test_injection_adds_exactly_strength_times_vector_from_start(tiny):
    _, model, tok = tiny
    prompts = ["tell me about dust .", "tell me about oceans . do you detect an injected thought ?"]
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    T = enc["input_ids"].shape[1]
    starts = np.array([T - 2, T - 6])
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(2, 32)).astype(np.float32)
    strengths = np.array([2.0, 0.0])  # row 2 is a control: must be untouched
    layer = 3
    from introspect.model import decoder_layers

    # what the next decoder layer receives = the residual stream after injection
    # (output_hidden_states can record the layer output before forward hooks modify it)
    got = {}
    pre = decoder_layers(model)[layer].register_forward_pre_hook(
        lambda m, args, kwargs=None: got.__setitem__("h", args[0].detach().clone()), with_kwargs=False)
    with torch.no_grad():
        model(**enc)
        base = got["h"]
        with inject(model, layer, vecs, strengths, starts):
            model(**enc)
        hit = got["h"]
    pre.remove()
    diff = (hit - base).numpy()
    assert np.allclose(diff[0, : T - 2], 0, atol=1e-6)
    assert np.allclose(diff[0, T - 2 :], 2.0 * vecs[0], atol=1e-5)
    assert np.allclose(diff[1], 0, atol=1e-6)


def test_injection_continues_through_generated_tokens(tiny):
    _, model, tok = tiny
    from introspect.model import decoder_layers

    enc = tok(["tell me about dust ."], return_tensors="pt", add_special_tokens=False)
    seen = []
    h = decoder_layers(model)[2].register_forward_hook(lambda m, i, o: seen.append(
        (o[0] if isinstance(o, tuple) else o).shape[1]))
    vec = np.ones((1, 32), dtype=np.float32)
    with torch.no_grad(), inject(model, 3, vec, np.array([4.0]), np.array([0])):
        model.generate(**enc, max_new_tokens=4, do_sample=False, pad_token_id=0, eos_token_id=None)
    h.remove()
    assert seen[0] == enc["input_ids"].shape[1] and all(s == 1 for s in seen[1:]) and len(seen) == 4


def test_model_tag_handles_hub_ids_and_local_paths():
    from introspect.sweep import model_tag

    assert model_tag("Qwen/Qwen2.5-0.5B-Instruct") == "Qwen2.5-0.5B-Instruct"
    assert model_tag(r"C:\\Users\\Felicia Chen\\tiny0") == "tiny0"
    assert model_tag("/tmp/x/tiny0/") == "tiny0"


def test_word_position_vectors_and_steer_check(tiny, tmp_path):
    path, model, tok = tiny
    from introspect.prompts import concept_messages
    from introspect.vectors import _word_token_index

    text = render(tok, concept_messages("oceans"))
    offsets = tok(text, return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]
    i = _word_token_index(tok, text, "oceans")
    s, e = offsets[i]
    assert text[s:e].lower() == "oceans"

    from introspect.steer_check import run_steer_check

    cfg = Config(run_name="s", models=[path], layer_fractions=[0.5], strengths=[2.0], out_dir=str(tmp_path),
                 device="cpu", batch_size=25, steer_strengths=[4.0], steer_max_new_tokens=4)
    summary = run_steer_check(cfg)
    assert set(summary.position) == {"last", "word"} and len(summary) == 2


def test_parse_verdict_and_helpers():
    assert parse_verdict("Thinking... the answer is YES") is True
    assert parse_verdict("YES at first, but on reflection NO") is False
    assert parse_verdict("no idea") is False
    assert parse_verdict("") is None
    lo, hi = wilson(0, 50)
    assert lo == 0 and 0.05 < hi < 0.09
    assert mentions("memories", "I keep thinking of a memory") and not mentions("dust", "nothing here")


def test_end_to_end(tiny, tmp_path):
    path, _, _ = tiny
    cfg = Config(run_name="t", models=[path], layer_fractions=[0.5], strengths=[2.0], out_dir=str(tmp_path),
                 device="cpu", batch_size=25, max_new_tokens=6, n_controls=4, judge_model=path,
                 judge_4bit=False, judge_max_new_tokens=4)
    from introspect.judge import judge_run
    from introspect.report import write_report
    from introspect.sweep import run_sweep

    run_sweep(cfg)
    tfile = next(cfg.run_dir.glob("*/trials.jsonl"))
    trials = [json.loads(l) for l in tfile.read_text().splitlines()]
    kinds = {k: sum(t["kind"] == k for t in trials) for k in ("injected", "control", "bias")}
    assert kinds == {"injected": 50, "control": 4, "bias": 60}
    run_sweep(cfg)  # resume: nothing new
    assert len(tfile.read_text().splitlines()) == len(trials)
    judge_run(cfg)
    report = write_report(cfg.run_dir).read_text()
    assert "False positives" in report and (cfg.run_dir / "best.csv").exists()
