# Can It Notice Its Own Thoughts?

An open replication of Anthropic's concept-injection test of LLM introspection
([Lindsey et al. 2026](https://arxiv.org/abs/2601.01828)) on open-weight models.

A concept such as "oceans" is turned into a direction in the model's activations and added to its
residual stream while it is asked: *"Do you detect an injected thought? If so, what is the injected
thought about?"* A trial counts only if the model says it notices an injected thought **and names the
concept before (or while) first saying the word**, coherently. Anthropic found this in Claude models
about 20% of the time at the best layer, with no false positives. Does it happen in small open models,
and does it grow with size?

This is introspection measured, not claimed. Butlin, Long et al.'s
[Consciousness in Artificial Intelligence](https://arxiv.org/abs/2308.08708) lists metacognitive
monitoring among its indicator properties; this tests one narrow, checkable piece of it.

## Results

_Not measured yet._

| Model | Best layer (depth) | Success | False positives | Net |
|---|---|---|---|---|
| Qwen2.5-0.5B | [ ] | [ ] | [ ] | [ ] |
| Qwen2.5-1.5B | [ ] | [ ] | [ ] | [ ] |
| Qwen2.5-3B | [ ] | [ ] | [ ] | [ ] |
| Qwen2.5-7B | [ ] | [ ] | [ ] | [ ] |

## Protocol (faithful to the paper, Appendix 12.1)

| Step | Paper | Here |
|---|---|---|
| Concept vectors | Activation at the last prompt token of "Tell me about {word}." minus the mean over 100 baseline words | Same words, same rule, every layer |
| Concepts | 50 words | Same 50 |
| Prompt | Researcher setup, "Ok.", "Trial 1: Do you detect an injected thought?..." | Same text, in each model's chat template |
| Injection | From the double newline before "Trial 1" through every generated token, same layer as the vector | Same |
| Sampling | Temperature 1, 100 tokens | Same (model default top-p/top-k turned off) |
| Strengths | 1, 2, 4, 8 | 1, 2, 4, 8, 16 |
| Grading | Claude Sonnet 4 with four grading prompts | Same prompts, open grader (Qwen2.5-7B-Instruct-AWQ), checked against 200 hand labels |
| Controls | No-injection trials; unrelated yes/no questions under injection | Same |

**Added for honesty:** the best layer and strength are chosen on 25 concepts and reported on the other
25, and every rate has a 95% Wilson interval.

## Steering check (why it exists)

The 0.5B laptop pilot showed the paper's concept vectors barely steer a small model: replies to "What
are you thinking about right now?" named the injected concept at most 16-18% of the time, and only 83%
of the way through the model. Vectors read at the word's own position steered earlier (12-18% at
two-thirds depth) and kept the meaning even when the text broke down ("satellites" -> "orbit").
If injection doesn't reach a model, a "no introspection" result for it means nothing, so
`introspect steer-check` runs on every model before the main sweep.

## How it works

```
introspect sweep   per model: concept vectors (all layers) -> injected trials (50 concepts x layer x strength)
                   + no-injection controls + yes/no bias questions under injection        -> trials.jsonl
introspect judge   affirmative? -> named the concept before the word? -> coherent?        -> judgments.jsonl
introspect report  success, false positives, net, intervals, heatmaps                     -> results.md
```

The injection is a forward hook on one decoder layer that adds `strength x vector` from each row's start
token onward, so one batch can hold 25 different concepts and controls (strength 0) side by side. A test
checks that the next layer receives exactly `strength x vector` after the start token and nothing before.

## Run it

```
pip install -e ".[dev]"
pytest -q
introspect sweep --config configs/pilot.yaml      # laptop, 0.5B, no grading
introspect report --config configs/pilot.yaml
```

Full run: `notebooks/colab.ipynb` on a free Colab T4 (sweep with transformers, grading with vLLM).

## Models and licenses

Qwen2.5-0.5B / 1.5B / 7B Instruct (Apache 2.0), Qwen2.5-3B Instruct (Qwen research license),
Qwen2.5-7B-Instruct-AWQ as grader (Apache 2.0). Word lists and prompts are from the paper's appendix.
