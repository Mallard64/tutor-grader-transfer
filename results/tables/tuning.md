## Validation comparison (selection metric -- test set not touched)

3 seeds per arm, math validation split (334 responses / 41 dialogues).
Arms differ from the control in exactly one field.

| arm | differs by | seed 0 | seed 1 | seed 2 | mean val macro-F1 |
|---|---|---|---|---|---|
| tune_a_base | control (4 epochs, unweighted) | 0.4894 | 0.5330 | 0.5097 | 0.5107 ± 0.0218 |
| tune_b_weighted | + balanced class weights | 0.5185 | 0.5647 | 0.5342 | **0.5391 ± 0.0235** |

Unpaired, the two overlap: the gap is 1.21 sd of either arm's own spread.
Paired by seed it does not, because every seed moves the same way and by
almost the same amount:

| seed | A | B | delta |
|---|---|---|---|
| 0 | 0.4894 | 0.5185 | +0.0291 |
| 1 | 0.5330 | 0.5647 | +0.0317 |
| 2 | 0.5097 | 0.5342 | +0.0245 |
| **mean** | | | **+0.0284 ± 0.0036** (paired t(2) = 13.5) |

Class weighting is worth about +0.028 validation macro-F1. Same seed, same
data order, one field changed -- the pairing is what makes 3 seeds enough here.

Arm C (8 epochs) was started and interrupted; it has no results. It is still
the motivated arm: in arm A the selected checkpoint was the *last* epoch on
every seed, which is what "4 epochs is too few" looks like. The untested
recipe is class weights AND more epochs together.
