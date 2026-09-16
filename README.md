# tutor-grader-transfer

## Question

The BEA 2025 shared task rates AI tutor responses on four pedagogical dimensions
(mistake identification, mistake location, providing guidance, actionability), each
3-class: `No` / `To some extent` / `Yes` — all of it grade-school math. To grade a
*programming* tutor, must you start over? This measures the drop, and how quickly a
few labeled programming examples recover it.

## Data

- **Math**: the BEA 2025 / MRBench dev set from
  [UnifyingAITutorEvaluation](https://github.com/kaushal0494/UnifyingAITutorEvaluation) —
  2,476 labeled responses, 300 dialogues, 9 tutors. The released test set is unlabeled,
  so the dev set is split three ways. Splits are **by dialogue** (8-9 tutors answer each
  one, so splitting by response would put the same context in train and test); dialogues
  sharing a source problem id (`292861665`, `292861665_1`) are grouped too.
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

k ∈ {0, 8, 16, 32, 64, 128}, 5 seeds each. The test split is fixed across seeds, so the
curve measures sample efficiency, not split noise.

Scoring (`eval/`): macro-F1 per dimension with 95% bootstrap CIs that resample whole
dialogues, ECE, and shortcut probes for length, code-block presence, and answer reveal.
Probes report the *gap* vs the same correlation on human labels — long responses really
do carry more guidance, so a raw correlation proves nothing.

## Results

Measured on the math test set (473 responses, 57 held-out dialogues):

| system | mean macro-F1 | MI | ML | PG | AC | ECE |
|---|---|---|---|---|---|---|
| (a) majority class | 0.253 | 0.295 | 0.260 | 0.235 | 0.223 | - |
| (a2) TF-IDF + logreg | **0.517** | 0.544 | 0.509 | 0.502 | 0.515 | 0.076 |

(a2) CIs are about ±0.06 (MI 0.544 [0.495, 0.604]); shortcut gaps stay small (worst
−0.12), so it tracks length slightly *less* than the human labels do. (c)/(d)/(e) are
unrun — they need hand-labeled programming data — and (b) costs API credit. The DeBERTa
code is tested end to end (`pytest -m slow`) but 4 epochs did not fit in 16GB RAM here.

## Limitations

- The domains differ in more than subject: math here is K-8 word problems, programming
  is college-level code. A drop in (c) spans subject *and* level, inseparably.
- Without reported inter-annotator agreement the programming labels are single-annotator
  and the ceiling on any grader is unknown.
- Programming responses are model-generated; real tutors are not that distribution.
- "Reveals the answer" is a keyword heuristic that over-fires: a probe, not a label.
- The math split is 70/15/15 by dialogue group; sizes vary, so counts land near
  67/13/19. (a2) is one seed; the 5-seed protocol covers the k-curves.
- DeBERTa needs real memory: batch 16 x 512 swapped on a 16GB laptop, ~30x slower.

## Run it

```bash
uv sync --extra train --extra llm --extra label
uv run python scripts/download_data.py          # BEA dev set -> data/raw/
uv run pytest                                   # 72 tests (5 need the train extra)
uv run python scripts/run_experiments.py a a2   # 'all' adds c, d, e (not b: costs money)
```

API keys (only for `build_candidates.py` and the judge) go in `.env` — see `.env.example`.
