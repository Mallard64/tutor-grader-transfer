# Results

macro-F1, mean +- std across seeds. Higher is better; ECE lower is better.
`shortcut gap` = Spearman(prediction, feature) - Spearman(human label, feature),
worst feature per run. Near zero means the model leans on surface features no
more than the human labels do.

| exp | system | eval domain | k | seeds | mean macro-F1 | MI | ML | PG | AC | ECE | shortcut gap |
|---|---|---|---|---|---|---|---|---|---|---|---|
| a | majority | math | 0 | 1 | 0.253 | 0.295 | 0.260 | 0.235 | 0.223 | - | - |
| a | majority | programming | 0 | 1 | 0.272 | 0.267 | 0.238 | 0.287 | 0.296 | - | - |
| a2 | tfidf_logreg | math | 0 | 1 | 0.517 | 0.544 | 0.509 | 0.502 | 0.515 | 0.076 | -0.122 |
| a2 | tfidf_logreg | programming | 0 | 1 | 0.533 | 0.508 | 0.397 | 0.656 | 0.570 | 0.282 | -0.347 |
| c | math_trained | math | 0 | 5 | 0.521 ± 0.015 | 0.603 | 0.480 | 0.481 | 0.519 | 0.046 | +0.146 |
| c | math_trained_cw | math | 0 | 1 | 0.534 | 0.617 | 0.525 | 0.495 | 0.500 | 0.091 | +0.182 |
| c | tfidf_math_trained | math | 0 | 1 | 0.517 | 0.544 | 0.509 | 0.502 | 0.515 | 0.076 | -0.122 |
| c | tfidf_math_trained | programming | 0 | 1 | 0.210 | 0.165 | 0.270 | 0.268 | 0.138 | 0.235 | -0.454 |
| d | tfidf_math_plus_k | programming | 0 | 5 | 0.185 ± 0.000 | 0.099 | 0.274 | 0.274 | 0.093 | 0.230 | -0.437 |
| d | tfidf_math_plus_k | programming | 8 | 5 | 0.295 ± 0.010 | 0.293 | 0.336 | 0.273 | 0.278 | 0.094 | -0.495 |
| d | tfidf_math_plus_k | programming | 16 | 5 | 0.301 ± 0.010 | 0.301 | 0.285 | 0.282 | 0.336 | 0.151 | -0.486 |
| d | tfidf_math_plus_k | programming | 32 | 5 | 0.302 ± 0.014 | 0.294 | 0.251 | 0.295 | 0.367 | 0.176 | -0.424 |
| d | tfidf_math_plus_k | programming | 64 | 5 | 0.301 ± 0.026 | 0.279 | 0.252 | 0.309 | 0.364 | 0.161 | -0.206 |
| d | tfidf_math_plus_k | programming | 128 | 5 | 0.306 ± 0.033 | 0.292 | 0.268 | 0.340 | 0.323 | 0.143 | -0.488 |
| e | tfidf_prog_only | programming | 0 | 5 | 0.147 ± 0.000 | 0.264 | 0.241 | 0.046 | 0.039 | 0.272 | - |
| e | tfidf_prog_only | programming | 8 | 5 | 0.258 ± 0.028 | 0.272 | 0.235 | 0.226 | 0.299 | 0.212 | -0.436 |
| e | tfidf_prog_only | programming | 16 | 5 | 0.275 ± 0.014 | 0.250 | 0.270 | 0.275 | 0.306 | 0.191 | -0.289 |
| e | tfidf_prog_only | programming | 32 | 5 | 0.308 ± 0.038 | 0.305 | 0.292 | 0.299 | 0.336 | 0.201 | -0.472 |
| e | tfidf_prog_only | programming | 64 | 5 | 0.360 ± 0.060 | 0.315 | 0.360 | 0.381 | 0.385 | 0.229 | -0.261 |
| e | tfidf_prog_only | programming | 128 | 5 | 0.459 ± 0.030 | 0.333 | 0.424 | 0.512 | 0.566 | 0.230 | -0.430 |
