# Results

macro-F1, mean +- std across seeds. Higher is better; ECE lower is better.
`shortcut gap` = Spearman(prediction, feature) - Spearman(human label, feature),
worst feature per run. Near zero means the model leans on surface features no
more than the human labels do.

| exp | system | eval domain | k | seeds | mean macro-F1 | MI | ML | PG | AC | ECE | shortcut gap |
|---|---|---|---|---|---|---|---|---|---|---|---|
| a | majority | math | 0 | 1 | 0.253 | 0.295 | 0.260 | 0.235 | 0.223 | - | - |
| a2 | tfidf_logreg | math | 0 | 1 | 0.517 | 0.544 | 0.509 | 0.502 | 0.515 | 0.076 | -0.122 |
| c | math_trained | math | 0 | 5 | 0.521 ± 0.015 | 0.603 | 0.480 | 0.481 | 0.519 | 0.046 | +0.146 |
| c | math_trained_cw | math | 0 | 1 | 0.534 | 0.617 | 0.525 | 0.495 | 0.500 | 0.091 | +0.182 |
| c | tfidf_math_trained | math | 0 | 1 | 0.517 | 0.544 | 0.509 | 0.502 | 0.515 | 0.076 | -0.122 |
| c | math_trained_cw | programming | 0 | 1 | 0.229 | 0.333 | 0.312 | 0.083 | 0.187 | 0.247 | -0.391 |
| c | tfidf_math_trained | programming | 0 | 1 | 0.229 | 0.200 | 0.389 | 0.254 | 0.073 | 0.248 | -0.691 |
| d | tfidf_math_plus_k | programming | 0 | 3 | 0.208 ± 0.000 | 0.121 | 0.422 | 0.222 | 0.067 | 0.217 | -0.388 |
| d | tfidf_math_plus_k | programming | 2 | 3 | 0.183 ± 0.063 | 0.212 | 0.167 | 0.210 | 0.144 | 0.242 | -0.648 |
| d | tfidf_math_plus_k | programming | 4 | 3 | 0.243 ± 0.058 | 0.135 | 0.392 | 0.237 | 0.207 | 0.258 | -0.355 |
| d | tfidf_math_plus_k | programming | 8 | 3 | 0.237 ± 0.045 | 0.205 | 0.167 | 0.333 | 0.242 | 0.259 | -0.429 |
| e | tfidf_prog_only | programming | 0 | 3 | 0.093 ± 0.000 | 0.205 | 0.167 | 0.000 | 0.000 | 0.194 | - |
| e | tfidf_prog_only | programming | 2 | 3 | 0.111 ± 0.047 | 0.242 | 0.200 | 0.000 | 0.000 | 0.246 | -0.457 |
| e | tfidf_prog_only | programming | 4 | 3 | 0.197 ± 0.092 | 0.205 | 0.216 | 0.165 | 0.202 | 0.179 | -0.477 |
| e | tfidf_prog_only | programming | 8 | 3 | 0.243 ± 0.000 | 0.205 | 0.167 | 0.279 | 0.320 | 0.299 | -0.782 |
