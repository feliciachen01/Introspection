"""introspect: Can it notice its own thoughts?

An open replication of the concept-injection test in Lindsey et al. (2026), "Emergent Introspective
Awareness in Large Language Models" (https://arxiv.org/abs/2601.01828), on open-weight models.

  sweep   concept vectors, then trials with a concept injected into the residual stream
  judge   grade replies with the paper's grading prompts and an open grader
  report  success, false positives and net rate per model, with intervals
"""

__version__ = "0.1.0"
