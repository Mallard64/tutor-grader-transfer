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
  2,476 labeled responses, 300 dialogues, 9 tutors. Its released test set is unlabeled, so
  the dev set is split three ways **by dialogue** (8-9 tutors answer each one, so splitting
  by response would leak context); ids sharing a problem are grouped too.
- **Programming**: Socratic Debugging Benchmark dialogues joined to annotations by
  `scripts/build_pilot.py`. The pilot is 20 LLM-labeled turns — enough to exercise the
  pipeline, not to conclude anything. `labeling/app.py` is the human path.

## Method

DeBERTa-v3-base, four 3-class heads on a shared encoder (`models/`), HF Trainer, one YAML
config per run in `configs/`. Graders: **(a)** majority class, **(a2)** TF-IDF + logistic
regression, **(b)** zero-shot LLM judge with the rubric. Transfer: **(c)** math-trained →
programming zero-shot, **(d)** math-trained + k programming examples, **(e)** programming
only. k ∈ {0, 8, 16, 32, 64, 128}, 5 seeds, test split fixed across seeds so the curve
measures sample efficiency rather than split noise. c/d/e exist for both the DeBERTa
(`transfer.py`, GPU) and TF-IDF (`linear_transfer.py`, CPU) graders.

Scoring (`eval/`): macro-F1 per dimension with 95% bootstrap CIs resampling whole dialogues,
ECE, and shortcut probes for length, code blocks, and answer reveal. Probes report the *gap*
vs the same correlation on human labels — long responses really do carry more guidance, so a
raw correlation proves nothing.

## Results

Math test set (473 responses / 57 dialogues); DeBERTa trained on a Colab T4.

| system | seeds | mean macro-F1 | MI | ML | PG | AC | ECE |
|---|---|---|---|---|---|---|---|
| (a) majority class | 1 | 0.253 | 0.295 | 0.260 | 0.235 | 0.223 | - |
| (a2) TF-IDF + logreg | 1 | 0.517 | 0.544 | 0.509 | 0.502 | 0.515 | 0.076 |
| (c) DeBERTa, unweighted | 5 | 0.521 ± 0.015 | 0.603 | 0.480 | 0.481 | 0.519 | 0.046 |
| (c) DeBERTa, class weights | 1 | **0.534** | 0.617 | 0.525 | 0.495 | 0.500 | 0.091 |

Class weighting was chosen on validation (+0.028, paired across 3 seeds,
`results/tables/tuning.md`) and holds up on test. Validation selects the last epoch
every time, so 4 epochs is too few; the 8-epoch arm is unrun.

**Transfer (pilot, 20 LLM-labeled programming responses):** both graders land on
**0.229**, between constant-prediction floors of 0.214 (majority from math) and
0.257 (majority of programming).

Identical means, different per-dimension profiles (TF-IDF is better on mistake
location, DeBERTa on mistake identification; see `results/tables/results.md`). TF-IDF's failure had an obvious
cause — 46% of programming tokens fall outside a vocabulary fitted on word problems
— but DeBERTa's subword tokenizer has no such wall and lands in the same place, so
vocabulary was not the whole story. Neither beats predicting a constant. **n=20,
CIs up to ±0.16, one seed, LLM labels: a pilot signal, not a result.** Two warnings:
the LLM labels correlate with response length far more than the human math labels do
(0.38–0.44 vs 0.12–0.26), and no programming response contains a code block, so that
probe is degenerate here.

## Limitations

- Domains differ in more than subject: math is K-8 word problems, programming is college
  code, so a drop in (c) spans subject *and* level, inseparably.
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
