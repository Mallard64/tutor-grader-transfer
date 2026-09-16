# Results

macro-F1, mean +- std across seeds. Higher is better; ECE lower is better.
`shortcut gap` = Spearman(prediction, feature) - Spearman(human label, feature),
worst feature per run. Near zero means the model leans on surface features no
more than the human labels do.

| exp | system | eval domain | k | seeds | mean macro-F1 | MI | ML | PG | AC | ECE | shortcut gap |
|---|---|---|---|---|---|---|---|---|---|---|---|
| a | majority | math | 0 | 1 | 0.253 | 0.295 | 0.260 | 0.235 | 0.223 | - | - |
| a2 | tfidf_logreg | math | 0 | 1 | 0.517 | 0.544 | 0.509 | 0.502 | 0.515 | 0.076 | -0.122 |
| c | math_trained | math | 0 | 1 | 0.506 | 0.595 | 0.468 | 0.443 | 0.516 | 0.055 | +0.135 |
