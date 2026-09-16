# tutor-grader-transfer

## Question

The BEA 2025 shared task rates AI tutor responses on four pedagogical dimensions
(mistake identification, mistake location, providing guidance, actionability), each
3-class: `No` / `To some extent` / `Yes` — all of it grade-school math. To grade a
*programming* tutor, must you start over? This measures the drop, and how quickly a few
labeled programming examples recover it.

## Data

- **Math**: the BEA 2025 / MRBench dev set from
  [UnifyingAITutorEvaluation](https://github.com/kaushal0494/UnifyingAITutorEvaluation) —
  2,476 labeled responses, 300 dialogues, 9 tutors. The released test set is unlabeled, so
  the dev set is split three ways, **by dialogue** (8-9 tutors answer each one, so
  splitting by response would leak context); ids sharing a problem are grouped too.
- **Programming**: not included. `scripts/build_candidates.py` turns TutorCode-format
  buggy code into candidates from 3+ models; you label them by hand in `labeling/app.py`.
  Nothing auto-labels with an LLM — that would make the comparison circular.

## Method

DeBERTa-v3-base, four 3-class heads on a shared encoder (`models/`), HF Trainer, one
YAML config per run in `configs/`. Experiments (`experiments/`):

| | system |
|---|---|
| a | majority class, fit on train |
| a2 | TF-IDF + logistic regression |
| b | zero-shot LLM judge, given the rubric, both domains |
| c | math-trained, evaluated on programming zero-shot |
| d | math-trained + k programming examples |
| e | programming-only, k examples, no math pretraining |

k ∈ {0, 8, 16, 32, 64, 128}, 5 seeds each; the test split is fixed across seeds, so the
curve measures sample efficiency, not split noise.

Scoring (`eval/`): macro-F1 per dimension with 95% bootstrap CIs resampling whole
dialogues, ECE, and shortcut probes for length, code blocks, and answer reveal. Probes
report the *gap* vs the same correlation on human labels — long responses really do carry
more guidance, so a raw correlation proves nothing.

## Results

| system | seeds | mean macro-F1 | MI | ML | PG | AC | ECE | shortcut gap |
|---|---|---|---|---|---|---|---|---|
| (a) majority class | 1 | 0.253 | 0.295 | 0.260 | 0.235 | 0.223 | - | - |
| (a2) TF-IDF + logreg | 1 | 0.517 | 0.544 | 0.509 | 0.502 | 0.515 | 0.076 | −0.12 |
| (c) DeBERTa-v3-base | 5 | 0.521 ± 0.015 | 0.603 | 0.480 | 0.481 | 0.519 | 0.046 | **+0.15** |

Math test set: 473 responses / 57 held-out dialogues; DeBERTa trained on a Colab T4.
Across 5 seeds DeBERTa spans 0.504–0.536 and ties the bag-of-words baseline (+0.004,
0.25 sd) rather than beating it. The two are good at different things: DeBERTa is far
better at mistake identification (0.603 vs 0.544), worse at guidance (0.481 vs 0.502),
and better calibrated (0.046 vs 0.076). But it tracks response length **more** than the
human labels do (+0.15) where the linear model tracks it less (−0.12): the accuracy is a
tie, the shortcut reliance is not. (d)/(e) need labeled programming data; (b) costs credit.

## Limitations

- Domains differ in more than subject: math is K-8 word problems, programming is
  college-level code, so a drop in (c) spans subject *and* level, inseparably.
- Without inter-annotator agreement the programming labels are single-annotator, so the
  ceiling on any grader is unknown.
- Programming responses are model-generated; real tutors are not that distribution, and
  "reveals the answer" is a keyword heuristic that over-fires: a probe, not a label.
- Split is 70/15/15 by dialogue group; sizes vary, so counts land near 67/13/19. (a) and
  (a2) are single-seed and deterministic; only (c) is swept.
- DeBERTa needs real memory: batch 16 x 512 swapped on a 16GB laptop, ~30x slower.

## Run it

```bash
uv sync --extra train --extra llm --extra label
uv run python scripts/download_data.py        # BEA dev set -> data/raw/
uv run pytest                                 # 72 tests (5 need the train extra)
uv run python scripts/run_experiments.py all  # a, a2, c, d, e (not b: costs money)
```

API keys (for `build_candidates.py` and the judge only) go in `.env`; see `.env.example`.
