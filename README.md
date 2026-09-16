# tutor-grader-transfer

Does a grader trained on math tutor responses also work on programming tutor responses,
and how many labeled programming examples does it take to close the gap?

## Question

The BEA 2025 shared task rates AI tutor responses on four pedagogical dimensions
(mistake identification, mistake location, providing guidance, actionability), each
3-class: `No` / `To some extent` / `Yes`. All of that data is grade-school math. If you
want to grade a *programming* tutor, do you have to start over? This repo measures the
drop, and how quickly a few labeled programming examples recover it.

## Data

- **Math**: the BEA 2025 / MRBench dev set from
  [UnifyingAITutorEvaluation](https://github.com/kaushal0494/UnifyingAITutorEvaluation) —
  2,476 labeled responses, 300 dialogues, 9 tutors. The released test set is unlabeled,
  so the dev set is split three ways here. Splits are **by dialogue**: 8-9 tutors answer
  each dialogue, so a response-level split would put the same context in train and test.
  Dialogues sharing a source problem id (`292861665`, `292861665_1`) are grouped too.
- **Programming**: not included. `scripts/build_candidates.py` turns TutorCode-format
  buggy student code into candidate responses from 3+ models; you label them by hand in
  `labeling/app.py`, which shows the same rubric text. Nothing here auto-labels with an
  LLM — the experiment compares graders *against* human labels, so model-generated
  labels would make it circular.

## Method

DeBERTa-v3-base with four 3-class heads on a shared encoder (`models/`), trained with
HF Trainer; one YAML config per run in `configs/`. Experiments (`experiments/`):

| | system |
|---|---|
| a | majority class, fit on train |
| b | zero-shot LLM judge, given the rubric, both domains |
| c | math-trained, evaluated on programming zero-shot |
| d | math-trained + k programming examples |
| e | programming-only, k examples, no math pretraining |

k ∈ {0, 8, 16, 32, 64, 128}, 5 seeds each. The test split is fixed across seeds, so the
learning curve measures sample efficiency rather than split noise.

Scoring (`eval/`): macro-F1 per dimension with 95% bootstrap CIs that resample whole
dialogues, expected calibration error, and shortcut probes for response length,
code-block presence, and whether the answer is revealed. The probes report the *gap*
against the same correlation on human labels — long responses genuinely do contain more
guidance, so a raw correlation is not by itself evidence of a shortcut.

## Results

Not yet run: (c), (d), (e) need the programming data, which requires hand-labeling, and
(b) costs API credit. The only measured number so far is the math majority baseline —
**mean macro-F1 0.253** (MI 0.295, ML 0.260, PG 0.235, AC 0.223) on 473 held-out
responses. Worth noting because `Yes` is 78% of mistake identification, so accuracy
would look far healthier than the task actually is.

`scripts/make_table.py` and `make_plots.py` fill `results/`; nothing is written by hand.

## Limitations

- The domains differ in more than subject: math here is K-8 word problems, programming
  is college-level code. A drop in (c) spans subject *and* level, inseparably.
- Without reported inter-annotator agreement the programming labels are
  single-annotator, and the ceiling on any grader is unknown.
- Programming responses are model-generated; real tutors are not that distribution.
- "Reveals the answer" is a keyword heuristic that over-fires: a probe feature, not a label.
- The math split is 70/15/15 by dialogue group; dialogues vary in size, so realized
  response counts land near 67/13/19.

## Run it

```bash
uv sync --extra train --extra llm --extra label
uv run python scripts/download_data.py          # BEA dev set -> data/raw/
uv run pytest                                   # 72 tests (5 need the train extra)
uv run python scripts/run_experiments.py a c    # 'all' runs a, c, d, e (not b: costs money)
```

API keys (only for `build_candidates.py` and the judge) go in `.env` — see `.env.example`.
