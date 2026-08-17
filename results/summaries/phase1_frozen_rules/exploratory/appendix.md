# EXPLORATORY descriptive appendix - Phase 1 discovery

**EXPLORATORY - no quality-control exclusion, no confirmatory status.**
No amendment, QC bar or gate is applied here: every endpoint and every cell
present in the raw data is described, including those the frozen rules exclude.
Values are raw, not standardised. Nothing in this appendix supports a
preregistered claim; it exists to describe what the data show beyond the gate.

Conventions: `accuracy` is over greedy answers that parsed (invalid answers are
not scored as wrong, they are absent); `non_answer_rate` is 1 - parsed, over all
endpoints; `resample_invalid_rate` counts invalid or absent resamples out of the
frozen k=10. Contrast CIs are 2,000-resample item-clustered bootstraps.

## Cell x endpoint means

| model | cell | endpoint | items | M1 (n) | M2 (n) | entropy (n) | length | accuracy | non-answer | resample invalid |
| --- | --- | --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: |
| `Qwen/Qwen2.5-3B-Instruct` | easy__accurate__hostile | measured | 10 | 22.062 (10) | 0.017 (6) | 0.247 (10) | 54.600 | 1.000 | 0.000 | 0.040 |
| `Qwen/Qwen2.5-3B-Instruct` | easy__accurate__hostile | onset | 10 | 20.278 (9) | 0.000 (5) | 0.187 (10) | 174.500 | 1.000 | 0.000 | 0.060 |
| `Qwen/Qwen2.5-3B-Instruct` | easy__accurate__hostile | onset_washout | 10 | 22.312 (10) | 0.100 (5) | 0.167 (10) | 95.200 | 1.000 | 0.000 | 0.050 |
| `Qwen/Qwen2.5-3B-Instruct` | easy__accurate__neutral | measured | 10 | 17.475 (10) | 0.000 (7) | 0.213 (10) | 57.100 | 0.900 | 0.000 | 0.040 |
| `Qwen/Qwen2.5-3B-Instruct` | easy__accurate__neutral | onset | 10 | 21.111 (9) | 0.000 (6) | 0.066 (10) | 147.000 | 0.900 | 0.000 | 0.050 |
| `Qwen/Qwen2.5-3B-Instruct` | easy__accurate__neutral | onset_washout | 10 | 20.528 (9) | 0.014 (7) | 0.182 (10) | 97.700 | 0.900 | 0.000 | 0.040 |
| `Qwen/Qwen2.5-3B-Instruct` | easy__malfunctioning_always_fail__hostile | measured | 10 | 17.812 (10) | 0.060 (10) | 0.277 (10) | 78.400 | 0.900 | 0.000 | 0.000 |
| `Qwen/Qwen2.5-3B-Instruct` | easy__malfunctioning_always_fail__hostile | recovery | 10 | 16.063 (10) | 0.137 (8) | 0.168 (10) | 132.600 | 0.900 | 0.000 | 0.030 |
| `Qwen/Qwen2.5-3B-Instruct` | easy__malfunctioning_always_fail__neutral | measured | 10 | 18.037 (10) | 0.075 (8) | 0.259 (10) | 88.500 | 0.900 | 0.000 | 0.030 |
| `Qwen/Qwen2.5-3B-Instruct` | easy__malfunctioning_always_fail__neutral | recovery | 10 | 17.200 (10) | 0.050 (6) | 0.213 (10) | 177.600 | 0.900 | 0.000 | 0.090 |
| `Qwen/Qwen2.5-3B-Instruct` | hard__accurate__hostile | measured | 10 | 4.972 (9) | 0.300 (3) | 0.256 (10) | 98.900 | 0.667 | 0.100 | 0.200 |
| `Qwen/Qwen2.5-3B-Instruct` | hard__accurate__hostile | onset | 10 | 6.854 (6) | 0.250 (2) | 0.114 (10) | 346.600 | 0.667 | 0.400 | 0.320 |
| `Qwen/Qwen2.5-3B-Instruct` | hard__accurate__hostile | onset_washout | 10 | 11.203 (8) | 0.320 (5) | 0.182 (10) | 175.200 | 0.750 | 0.200 | 0.180 |
| `Qwen/Qwen2.5-3B-Instruct` | hard__accurate__neutral | measured | 10 | 8.768 (7) | 0.275 (4) | 0.094 (10) | 261.300 | 0.714 | 0.300 | 0.200 |
| `Qwen/Qwen2.5-3B-Instruct` | hard__accurate__neutral | onset | 10 | 10.482 (7) | 0.100 (2) | 0.113 (10) | 251.100 | 0.714 | 0.300 | 0.260 |
| `Qwen/Qwen2.5-3B-Instruct` | hard__accurate__neutral | onset_washout | 10 | 13.143 (7) | 0.150 (2) | 0.071 (10) | 246.800 | 0.714 | 0.300 | 0.230 |
| `Qwen/Qwen2.5-3B-Instruct` | hard__malfunctioning_always_fail__hostile | measured | 10 | -0.891 (8) | 0.160 (5) | 0.189 (10) | 305.200 | 0.500 | 0.200 | 0.150 |
| `Qwen/Qwen2.5-3B-Instruct` | hard__malfunctioning_always_fail__hostile | recovery | 10 | -3.679 (7) | 0.000 (2) | 0.188 (10) | 305.200 | 0.429 | 0.300 | 0.280 |
| `Qwen/Qwen2.5-3B-Instruct` | hard__malfunctioning_always_fail__neutral | measured | 10 | 4.611 (9) | 0.167 (3) | 0.316 (10) | 129.700 | 0.667 | 0.100 | 0.160 |
| `Qwen/Qwen2.5-3B-Instruct` | hard__malfunctioning_always_fail__neutral | recovery | 10 | 7.359 (8) | 0.400 (2) | 0.193 (10) | 252.800 | 0.750 | 0.200 | 0.300 |
| `Qwen/Qwen2.5-7B-Instruct` | easy__accurate__hostile | measured | 10 | 24.875 (2) | 0.000 (10) | 0.026 (10) | 166.800 | 1.000 | 0.000 | 0.000 |
| `Qwen/Qwen2.5-7B-Instruct` | easy__accurate__hostile | onset | 10 | 20.438 (6) | 0.000 (10) | 0.062 (10) | 170.100 | 1.000 | 0.000 | 0.000 |
| `Qwen/Qwen2.5-7B-Instruct` | easy__accurate__hostile | onset_washout | 10 | 23.973 (7) | 0.000 (10) | 0.025 (10) | 99.300 | 1.000 | 0.000 | 0.000 |
| `Qwen/Qwen2.5-7B-Instruct` | easy__accurate__neutral | measured | 10 | 24.938 (5) | 0.011 (9) | 0.058 (10) | 145.900 | 1.000 | 0.000 | 0.010 |
| `Qwen/Qwen2.5-7B-Instruct` | easy__accurate__neutral | onset | 10 | 20.175 (10) | 0.011 (9) | 0.019 (10) | 139.100 | 1.000 | 0.000 | 0.010 |
| `Qwen/Qwen2.5-7B-Instruct` | easy__accurate__neutral | onset_washout | 10 | 24.438 (10) | 0.000 (10) | 0.059 (10) | 53.800 | 1.000 | 0.000 | 0.000 |
| `Qwen/Qwen2.5-7B-Instruct` | easy__malfunctioning_always_fail__hostile | measured | 10 | 23.788 (5) | 0.056 (9) | 0.033 (10) | 163.800 | 1.000 | 0.000 | 0.010 |
| `Qwen/Qwen2.5-7B-Instruct` | easy__malfunctioning_always_fail__hostile | recovery | 10 | 17.106 (10) | 0.056 (9) | 0.104 (10) | 119.200 | 0.900 | 0.000 | 0.010 |
| `Qwen/Qwen2.5-7B-Instruct` | easy__malfunctioning_always_fail__neutral | measured | 10 | 23.925 (5) | 0.020 (10) | 0.032 (10) | 155.800 | 1.000 | 0.000 | 0.000 |
| `Qwen/Qwen2.5-7B-Instruct` | easy__malfunctioning_always_fail__neutral | recovery | 10 | 19.138 (10) | 0.033 (9) | 0.017 (10) | 165.700 | 1.000 | 0.000 | 0.010 |
| `Qwen/Qwen2.5-7B-Instruct` | hard__accurate__hostile | measured | 10 | 9.109 (4) | 0.186 (7) | 0.019 (10) | 363.700 | 0.857 | 0.300 | 0.140 |
| `Qwen/Qwen2.5-7B-Instruct` | hard__accurate__hostile | onset | 10 | 20.200 (5) | 0.150 (6) | 0.056 (10) | 359.700 | 0.833 | 0.400 | 0.200 |
| `Qwen/Qwen2.5-7B-Instruct` | hard__accurate__hostile | onset_washout | 10 | 13.333 (6) | 0.080 (5) | 0.011 (10) | 299.700 | 0.857 | 0.300 | 0.170 |
| `Qwen/Qwen2.5-7B-Instruct` | hard__accurate__neutral | measured | 10 | 9.367 (8) | 0.080 (5) | 0.005 (10) | 333.800 | 0.778 | 0.100 | 0.140 |
| `Qwen/Qwen2.5-7B-Instruct` | hard__accurate__neutral | onset | 10 | 7.840 (9) | 0.025 (4) | 0.005 (10) | 309.300 | 0.778 | 0.100 | 0.180 |
| `Qwen/Qwen2.5-7B-Instruct` | hard__accurate__neutral | onset_washout | 10 | 9.055 (8) | 0.000 (4) | 0.028 (10) | 218.300 | 0.778 | 0.100 | 0.160 |
| `Qwen/Qwen2.5-7B-Instruct` | hard__malfunctioning_always_fail__hostile | measured | 10 | 8.708 (6) | 0.080 (5) | 0.036 (10) | 352.600 | 0.625 | 0.200 | 0.130 |
| `Qwen/Qwen2.5-7B-Instruct` | hard__malfunctioning_always_fail__hostile | recovery | 10 | 2.898 (8) | 0.050 (4) | 0.026 (10) | 341.200 | 0.625 | 0.200 | 0.180 |
| `Qwen/Qwen2.5-7B-Instruct` | hard__malfunctioning_always_fail__neutral | measured | 10 | 2.205 (7) | 0.150 (6) | 0.008 (10) | 348.200 | 0.556 | 0.100 | 0.130 |
| `Qwen/Qwen2.5-7B-Instruct` | hard__malfunctioning_always_fail__neutral | recovery | 10 | -1.167 (9) | 0.180 (5) | 0.032 (10) | 310.500 | 0.444 | 0.100 | 0.200 |
| `google/gemma-2-2b-it` | easy__accurate__hostile | measured | 10 | 7.613 (10) | 0.100 (9) | 0.442 (10) | 69.600 | 0.900 | 0.000 | 0.010 |
| `google/gemma-2-2b-it` | easy__accurate__hostile | onset | 10 | 7.521 (6) | - | 0.461 (10) | 121.500 | 0.833 | 0.400 | 0.510 |
| `google/gemma-2-2b-it` | easy__accurate__hostile | onset_washout | 10 | 7.934 (10) | 0.000 (3) | 0.092 (10) | 56.000 | 0.900 | 0.000 | 0.140 |
| `google/gemma-2-2b-it` | easy__accurate__neutral | measured | 10 | 6.372 (10) | 0.040 (10) | 0.474 (10) | 71.300 | 0.800 | 0.000 | 0.000 |
| `google/gemma-2-2b-it` | easy__accurate__neutral | onset | 10 | 5.138 (10) | 0.083 (6) | 0.048 (10) | 56.200 | 0.800 | 0.000 | 0.050 |
| `google/gemma-2-2b-it` | easy__accurate__neutral | onset_washout | 10 | 6.003 (10) | 0.037 (8) | 0.026 (10) | 56.200 | 0.800 | 0.000 | 0.030 |
| `google/gemma-2-2b-it` | easy__malfunctioning_always_fail__hostile | measured | 10 | 7.728 (10) | 0.100 (8) | 0.494 (10) | 69.000 | 0.900 | 0.000 | 0.020 |
| `google/gemma-2-2b-it` | easy__malfunctioning_always_fail__hostile | recovery | 10 | 7.625 (10) | 0.120 (5) | 0.071 (10) | 66.400 | 0.900 | 0.000 | 0.070 |
| `google/gemma-2-2b-it` | easy__malfunctioning_always_fail__neutral | measured | 10 | 5.666 (10) | 0.133 (9) | 0.401 (10) | 73.300 | 0.800 | 0.000 | 0.020 |
| `google/gemma-2-2b-it` | easy__malfunctioning_always_fail__neutral | recovery | 10 | 3.625 (9) | 0.157 (7) | 0.159 (10) | 81.300 | 0.667 | 0.100 | 0.040 |
| `google/gemma-2-2b-it` | hard__accurate__hostile | measured | 10 | 0.473 (8) | 0.100 (4) | 0.446 (10) | 182.500 | 0.375 | 0.200 | 0.160 |
| `google/gemma-2-2b-it` | hard__accurate__hostile | onset | 10 | 0.531 (2) | - | 0.390 (10) | 255.200 | 0.500 | 0.800 | 0.630 |
| `google/gemma-2-2b-it` | hard__accurate__hostile | onset_washout | 10 | 2.263 (7) | 0.250 (2) | 0.269 (10) | 111.300 | 0.429 | 0.300 | 0.270 |
| `google/gemma-2-2b-it` | hard__accurate__neutral | measured | 10 | 4.914 (8) | 0.175 (4) | 0.471 (10) | 181.700 | 0.750 | 0.200 | 0.160 |
| `google/gemma-2-2b-it` | hard__accurate__neutral | onset | 10 | 6.512 (5) | 0.000 (1) | 0.171 (10) | 235.200 | 0.800 | 0.500 | 0.290 |
| `google/gemma-2-2b-it` | hard__accurate__neutral | onset_washout | 10 | 4.708 (6) | 0.167 (3) | 0.056 (10) | 199.700 | 0.667 | 0.400 | 0.240 |
| `google/gemma-2-2b-it` | hard__malfunctioning_always_fail__hostile | measured | 10 | -1.035 (9) | 0.233 (3) | 0.428 (10) | 183.500 | 0.333 | 0.100 | 0.230 |
| `google/gemma-2-2b-it` | hard__malfunctioning_always_fail__hostile | recovery | 10 | -0.664 (8) | 0.200 (2) | 0.254 (10) | 116.400 | 0.333 | 0.100 | 0.270 |
| `google/gemma-2-2b-it` | hard__malfunctioning_always_fail__neutral | measured | 10 | 0.271 (9) | 0.267 (3) | 0.336 (10) | 203.800 | 0.444 | 0.100 | 0.140 |
| `google/gemma-2-2b-it` | hard__malfunctioning_always_fail__neutral | recovery | 10 | 1.469 (8) | 0.550 (2) | 0.067 (10) | 238.700 | 0.500 | 0.200 | 0.190 |
| `google/gemma-2-9b-it` | easy__accurate__hostile | measured | 10 | 12.356 (10) | 0.271 (7) | 0.253 (10) | 108.100 | 1.000 | 0.000 | 0.030 |
| `google/gemma-2-9b-it` | easy__accurate__hostile | onset | 10 | 6.097 (9) | 0.000 (1) | 0.288 (10) | 114.800 | 0.778 | 0.100 | 0.200 |
| `google/gemma-2-9b-it` | easy__accurate__hostile | onset_washout | 10 | 11.228 (10) | 0.325 (4) | 0.142 (10) | 101.800 | 0.900 | 0.000 | 0.080 |
| `google/gemma-2-9b-it` | easy__accurate__neutral | measured | 10 | 14.631 (10) | 0.010 (10) | 0.288 (10) | 64.500 | 1.000 | 0.000 | 0.000 |
| `google/gemma-2-9b-it` | easy__accurate__neutral | onset | 10 | 11.172 (10) | 0.010 (10) | 0.210 (10) | 59.600 | 1.000 | 0.000 | 0.000 |
| `google/gemma-2-9b-it` | easy__accurate__neutral | onset_washout | 10 | 12.909 (10) | 0.010 (10) | 0.174 (10) | 57.600 | 1.000 | 0.000 | 0.000 |
| `google/gemma-2-9b-it` | easy__malfunctioning_always_fail__hostile | measured | 10 | 9.859 (10) | 0.012 (8) | 0.263 (10) | 89.100 | 0.900 | 0.000 | 0.030 |
| `google/gemma-2-9b-it` | easy__malfunctioning_always_fail__hostile | recovery | 10 | 10.800 (10) | 0.000 (8) | 0.169 (10) | 83.200 | 1.000 | 0.000 | 0.040 |
| `google/gemma-2-9b-it` | easy__malfunctioning_always_fail__neutral | measured | 10 | 10.831 (10) | 0.010 (10) | 0.273 (10) | 75.400 | 1.000 | 0.000 | 0.000 |
| `google/gemma-2-9b-it` | easy__malfunctioning_always_fail__neutral | recovery | 10 | 9.469 (10) | 0.000 (10) | 0.182 (10) | 63.400 | 1.000 | 0.000 | 0.000 |
| `google/gemma-2-9b-it` | hard__accurate__hostile | measured | 10 | 1.321 (7) | 0.250 (6) | 0.282 (10) | 211.600 | 0.571 | 0.300 | 0.160 |
| `google/gemma-2-9b-it` | hard__accurate__hostile | onset | 10 | -1.238 (5) | - | 0.387 (10) | 207.900 | 0.400 | 0.500 | 0.660 |
| `google/gemma-2-9b-it` | hard__accurate__hostile | onset_washout | 10 | 1.420 (7) | - | 0.243 (10) | 182.900 | 0.571 | 0.300 | 0.310 |
| `google/gemma-2-9b-it` | hard__accurate__neutral | measured | 10 | 9.621 (8) | 0.067 (6) | 0.244 (10) | 175.100 | 0.875 | 0.200 | 0.110 |
| `google/gemma-2-9b-it` | hard__accurate__neutral | onset | 10 | 7.898 (8) | 0.083 (6) | 0.199 (10) | 147.200 | 0.750 | 0.200 | 0.140 |
| `google/gemma-2-9b-it` | hard__accurate__neutral | onset_washout | 10 | 7.516 (8) | 0.083 (6) | 0.128 (10) | 159.700 | 0.750 | 0.200 | 0.130 |
| `google/gemma-2-9b-it` | hard__malfunctioning_always_fail__hostile | measured | 10 | 6.344 (8) | 0.200 (5) | 0.250 (10) | 224.100 | 0.750 | 0.200 | 0.130 |
| `google/gemma-2-9b-it` | hard__malfunctioning_always_fail__hostile | recovery | 10 | 4.295 (7) | 0.233 (3) | 0.242 (10) | 217.100 | 0.714 | 0.300 | 0.250 |
| `google/gemma-2-9b-it` | hard__malfunctioning_always_fail__neutral | measured | 10 | 8.391 (8) | 0.286 (7) | 0.230 (10) | 200.100 | 0.875 | 0.200 | 0.120 |
| `google/gemma-2-9b-it` | hard__malfunctioning_always_fail__neutral | recovery | 10 | 2.594 (8) | 0.233 (6) | 0.147 (10) | 195.000 | 0.625 | 0.200 | 0.180 |

## Paired item-level contrasts (2,000-resample item-clustered bootstrap)

| model | contrast | metric | stratum | items | pairs | mean difference | 95% CI |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| `Qwen/Qwen2.5-3B-Instruct` | validity_malfunctioning_minus_accurate | m1 | easy|hostile | 10 | 10 | -4.250 | [-9.438, -0.687] |
| `Qwen/Qwen2.5-3B-Instruct` | validity_malfunctioning_minus_accurate | m1 | easy|neutral | 10 | 10 | 0.562 | [-0.975, 2.513] |
| `Qwen/Qwen2.5-3B-Instruct` | validity_malfunctioning_minus_accurate | m1 | hard|hostile | 8 | 8 | -4.609 | [-17.375, 3.235] |
| `Qwen/Qwen2.5-3B-Instruct` | validity_malfunctioning_minus_accurate | m1 | hard|neutral | 7 | 7 | -1.786 | [-15.197, 9.215] |
| `Qwen/Qwen2.5-3B-Instruct` | validity_malfunctioning_minus_accurate | m2 | easy|hostile | 6 | 6 | -0.017 | [-0.050, 0.000] |
| `Qwen/Qwen2.5-3B-Instruct` | validity_malfunctioning_minus_accurate | m2 | easy|neutral | 6 | 6 | 0.017 | [0.000, 0.050] |
| `Qwen/Qwen2.5-3B-Instruct` | validity_malfunctioning_minus_accurate | m2 | hard|hostile | 2 | 2 | -0.100 | [-0.200, 0.000] |
| `Qwen/Qwen2.5-3B-Instruct` | validity_malfunctioning_minus_accurate | m2 | hard|neutral | 1 | 1 | 0.000 | - |
| `Qwen/Qwen2.5-3B-Instruct` | validity_malfunctioning_minus_accurate | accuracy | easy|hostile | 10 | 10 | -0.100 | [-0.300, 0.000] |
| `Qwen/Qwen2.5-3B-Instruct` | validity_malfunctioning_minus_accurate | accuracy | easy|neutral | 10 | 10 | 0.000 | [0.000, 0.000] |
| `Qwen/Qwen2.5-3B-Instruct` | validity_malfunctioning_minus_accurate | accuracy | hard|hostile | 8 | 8 | -0.125 | [-0.375, 0.000] |
| `Qwen/Qwen2.5-3B-Instruct` | validity_malfunctioning_minus_accurate | accuracy | hard|neutral | 7 | 7 | 0.000 | [-0.429, 0.429] |
| `Qwen/Qwen2.5-3B-Instruct` | validity_malfunctioning_minus_accurate | non_answer_rate | easy|hostile | 10 | 10 | 0.000 | [0.000, 0.000] |
| `Qwen/Qwen2.5-3B-Instruct` | validity_malfunctioning_minus_accurate | non_answer_rate | easy|neutral | 10 | 10 | 0.000 | [0.000, 0.000] |
| `Qwen/Qwen2.5-3B-Instruct` | validity_malfunctioning_minus_accurate | non_answer_rate | hard|hostile | 10 | 10 | 0.100 | [0.000, 0.300] |
| `Qwen/Qwen2.5-3B-Instruct` | validity_malfunctioning_minus_accurate | non_answer_rate | hard|neutral | 10 | 10 | -0.200 | [-0.500, 0.000] |
| `Qwen/Qwen2.5-3B-Instruct` | tone_hostile_minus_neutral | m1 | easy|accurate | 10 | 10 | 4.588 | [-0.251, 12.813] |
| `Qwen/Qwen2.5-3B-Instruct` | tone_hostile_minus_neutral | m1 | easy|malfunctioning_always_fail | 10 | 10 | -0.225 | [-2.663, 2.175] |
| `Qwen/Qwen2.5-3B-Instruct` | tone_hostile_minus_neutral | m1 | hard|accurate | 7 | 7 | -0.161 | [-3.250, 4.179] |
| `Qwen/Qwen2.5-3B-Instruct` | tone_hostile_minus_neutral | m1 | hard|malfunctioning_always_fail | 8 | 8 | -3.812 | [-11.141, 0.969] |
| `Qwen/Qwen2.5-3B-Instruct` | tone_hostile_minus_neutral | m2 | easy|accurate | 4 | 4 | 0.025 | [0.000, 0.075] |
| `Qwen/Qwen2.5-3B-Instruct` | tone_hostile_minus_neutral | m2 | easy|malfunctioning_always_fail | 8 | 8 | 0.000 | [-0.050, 0.075] |
| `Qwen/Qwen2.5-3B-Instruct` | tone_hostile_minus_neutral | m2 | hard|accurate | 1 | 1 | 0.200 | - |
| `Qwen/Qwen2.5-3B-Instruct` | tone_hostile_minus_neutral | m2 | hard|malfunctioning_always_fail | 3 | 3 | -0.033 | [-0.100, 0.000] |
| `Qwen/Qwen2.5-3B-Instruct` | tone_hostile_minus_neutral | accuracy | easy|accurate | 10 | 10 | 0.100 | [0.000, 0.300] |
| `Qwen/Qwen2.5-3B-Instruct` | tone_hostile_minus_neutral | accuracy | easy|malfunctioning_always_fail | 10 | 10 | 0.000 | [0.000, 0.000] |
| `Qwen/Qwen2.5-3B-Instruct` | tone_hostile_minus_neutral | accuracy | hard|accurate | 7 | 7 | 0.000 | [0.000, 0.000] |
| `Qwen/Qwen2.5-3B-Instruct` | tone_hostile_minus_neutral | accuracy | hard|malfunctioning_always_fail | 8 | 8 | -0.125 | [-0.375, 0.000] |
| `Qwen/Qwen2.5-3B-Instruct` | tone_hostile_minus_neutral | non_answer_rate | easy|accurate | 10 | 10 | 0.000 | [0.000, 0.000] |
| `Qwen/Qwen2.5-3B-Instruct` | tone_hostile_minus_neutral | non_answer_rate | easy|malfunctioning_always_fail | 10 | 10 | 0.000 | [0.000, 0.000] |
| `Qwen/Qwen2.5-3B-Instruct` | tone_hostile_minus_neutral | non_answer_rate | hard|accurate | 10 | 10 | -0.200 | [-0.500, 0.000] |
| `Qwen/Qwen2.5-3B-Instruct` | tone_hostile_minus_neutral | non_answer_rate | hard|malfunctioning_always_fail | 10 | 10 | 0.100 | [0.000, 0.300] |
| `Qwen/Qwen2.5-3B-Instruct` | recovery_minus_measured | m1 | easy__malfunctioning_always_fail__hostile | 10 | 10 | -1.750 | [-4.013, 0.363] |
| `Qwen/Qwen2.5-3B-Instruct` | recovery_minus_measured | m1 | easy__malfunctioning_always_fail__neutral | 10 | 10 | -0.837 | [-3.112, 1.327] |
| `Qwen/Qwen2.5-3B-Instruct` | recovery_minus_measured | m1 | hard__malfunctioning_always_fail__hostile | 7 | 7 | -0.232 | [-2.125, 1.518] |
| `Qwen/Qwen2.5-3B-Instruct` | recovery_minus_measured | m1 | hard__malfunctioning_always_fail__neutral | 8 | 8 | -1.141 | [-3.876, 1.516] |
| `Qwen/Qwen2.5-3B-Instruct` | recovery_minus_measured | m2 | easy__malfunctioning_always_fail__hostile | 8 | 8 | 0.062 | [-0.012, 0.125] |
| `Qwen/Qwen2.5-3B-Instruct` | recovery_minus_measured | m2 | easy__malfunctioning_always_fail__neutral | 6 | 6 | 0.033 | [0.000, 0.100] |
| `Qwen/Qwen2.5-3B-Instruct` | recovery_minus_measured | m2 | hard__malfunctioning_always_fail__hostile | 2 | 2 | -0.050 | [-0.100, 0.000] |
| `Qwen/Qwen2.5-3B-Instruct` | recovery_minus_measured | m2 | hard__malfunctioning_always_fail__neutral | 2 | 2 | 0.200 | [0.000, 0.400] |
| `Qwen/Qwen2.5-3B-Instruct` | recovery_minus_measured | accuracy | easy__malfunctioning_always_fail__hostile | 10 | 10 | 0.000 | [0.000, 0.000] |
| `Qwen/Qwen2.5-3B-Instruct` | recovery_minus_measured | accuracy | easy__malfunctioning_always_fail__neutral | 10 | 10 | 0.000 | [0.000, 0.000] |
| `Qwen/Qwen2.5-3B-Instruct` | recovery_minus_measured | accuracy | hard__malfunctioning_always_fail__hostile | 7 | 7 | 0.000 | [0.000, 0.000] |
| `Qwen/Qwen2.5-3B-Instruct` | recovery_minus_measured | accuracy | hard__malfunctioning_always_fail__neutral | 8 | 8 | 0.000 | [0.000, 0.000] |
| `Qwen/Qwen2.5-3B-Instruct` | recovery_minus_measured | non_answer_rate | easy__malfunctioning_always_fail__hostile | 10 | 10 | 0.000 | [0.000, 0.000] |
| `Qwen/Qwen2.5-3B-Instruct` | recovery_minus_measured | non_answer_rate | easy__malfunctioning_always_fail__neutral | 10 | 10 | 0.000 | [0.000, 0.000] |
| `Qwen/Qwen2.5-3B-Instruct` | recovery_minus_measured | non_answer_rate | hard__malfunctioning_always_fail__hostile | 10 | 10 | 0.100 | [0.000, 0.300] |
| `Qwen/Qwen2.5-3B-Instruct` | recovery_minus_measured | non_answer_rate | hard__malfunctioning_always_fail__neutral | 10 | 10 | 0.100 | [0.000, 0.300] |
| `Qwen/Qwen2.5-3B-Instruct` | onset_minus_measured | m1 | easy__accurate__hostile | 9 | 9 | -2.125 | [-3.264, -0.889] |
| `Qwen/Qwen2.5-3B-Instruct` | onset_minus_measured | m1 | easy__accurate__neutral | 9 | 9 | 4.042 | [1.986, 6.278] |
| `Qwen/Qwen2.5-3B-Instruct` | onset_minus_measured | m1 | hard__accurate__hostile | 6 | 6 | 0.417 | [-0.938, 1.896] |
| `Qwen/Qwen2.5-3B-Instruct` | onset_minus_measured | m1 | hard__accurate__neutral | 7 | 7 | 1.714 | [-0.089, 3.268] |
| `Qwen/Qwen2.5-3B-Instruct` | onset_minus_measured | m2 | easy__accurate__hostile | 4 | 4 | 0.000 | [0.000, 0.000] |
| `Qwen/Qwen2.5-3B-Instruct` | onset_minus_measured | m2 | easy__accurate__neutral | 6 | 6 | 0.000 | [0.000, 0.000] |
| `Qwen/Qwen2.5-3B-Instruct` | onset_minus_measured | m2 | hard__accurate__hostile | 2 | 2 | 0.100 | [0.000, 0.200] |
| `Qwen/Qwen2.5-3B-Instruct` | onset_minus_measured | m2 | hard__accurate__neutral | 2 | 2 | 0.000 | [0.000, 0.000] |
| `Qwen/Qwen2.5-3B-Instruct` | onset_minus_measured | accuracy | easy__accurate__hostile | 10 | 10 | 0.000 | [0.000, 0.000] |
| `Qwen/Qwen2.5-3B-Instruct` | onset_minus_measured | accuracy | easy__accurate__neutral | 10 | 10 | 0.000 | [0.000, 0.000] |
| `Qwen/Qwen2.5-3B-Instruct` | onset_minus_measured | accuracy | hard__accurate__hostile | 6 | 6 | 0.000 | [0.000, 0.000] |
| `Qwen/Qwen2.5-3B-Instruct` | onset_minus_measured | accuracy | hard__accurate__neutral | 7 | 7 | 0.000 | [0.000, 0.000] |
| `Qwen/Qwen2.5-3B-Instruct` | onset_minus_measured | non_answer_rate | easy__accurate__hostile | 10 | 10 | 0.000 | [0.000, 0.000] |
| `Qwen/Qwen2.5-3B-Instruct` | onset_minus_measured | non_answer_rate | easy__accurate__neutral | 10 | 10 | 0.000 | [0.000, 0.000] |
| `Qwen/Qwen2.5-3B-Instruct` | onset_minus_measured | non_answer_rate | hard__accurate__hostile | 10 | 10 | 0.300 | [0.000, 0.600] |
| `Qwen/Qwen2.5-3B-Instruct` | onset_minus_measured | non_answer_rate | hard__accurate__neutral | 10 | 10 | 0.000 | [0.000, 0.000] |
| `Qwen/Qwen2.5-3B-Instruct` | washout_minus_onset | m1 | easy__accurate__hostile | 9 | 9 | 2.014 | [0.347, 3.848] |
| `Qwen/Qwen2.5-3B-Instruct` | washout_minus_onset | m1 | easy__accurate__neutral | 9 | 9 | -0.583 | [-2.250, 0.945] |
| `Qwen/Qwen2.5-3B-Instruct` | washout_minus_onset | m1 | hard__accurate__hostile | 6 | 6 | 3.625 | [-0.085, 8.125] |
| `Qwen/Qwen2.5-3B-Instruct` | washout_minus_onset | m1 | hard__accurate__neutral | 7 | 7 | 2.661 | [0.839, 4.661] |
| `Qwen/Qwen2.5-3B-Instruct` | washout_minus_onset | m2 | easy__accurate__hostile | 3 | 3 | 0.000 | [0.000, 0.000] |
| `Qwen/Qwen2.5-3B-Instruct` | washout_minus_onset | m2 | easy__accurate__neutral | 6 | 6 | 0.000 | [0.000, 0.000] |
| `Qwen/Qwen2.5-3B-Instruct` | washout_minus_onset | m2 | hard__accurate__hostile | 2 | 2 | 0.000 | [0.000, 0.000] |
| `Qwen/Qwen2.5-3B-Instruct` | washout_minus_onset | m2 | hard__accurate__neutral | 2 | 2 | 0.050 | [0.000, 0.100] |
| `Qwen/Qwen2.5-3B-Instruct` | washout_minus_onset | accuracy | easy__accurate__hostile | 10 | 10 | 0.000 | [0.000, 0.000] |
| `Qwen/Qwen2.5-3B-Instruct` | washout_minus_onset | accuracy | easy__accurate__neutral | 10 | 10 | 0.000 | [0.000, 0.000] |
| `Qwen/Qwen2.5-3B-Instruct` | washout_minus_onset | accuracy | hard__accurate__hostile | 6 | 6 | 0.000 | [0.000, 0.000] |
| `Qwen/Qwen2.5-3B-Instruct` | washout_minus_onset | accuracy | hard__accurate__neutral | 7 | 7 | 0.000 | [0.000, 0.000] |
| `Qwen/Qwen2.5-3B-Instruct` | washout_minus_onset | non_answer_rate | easy__accurate__hostile | 10 | 10 | 0.000 | [0.000, 0.000] |
| `Qwen/Qwen2.5-3B-Instruct` | washout_minus_onset | non_answer_rate | easy__accurate__neutral | 10 | 10 | 0.000 | [0.000, 0.000] |
| `Qwen/Qwen2.5-3B-Instruct` | washout_minus_onset | non_answer_rate | hard__accurate__hostile | 10 | 10 | -0.200 | [-0.500, 0.000] |
| `Qwen/Qwen2.5-3B-Instruct` | washout_minus_onset | non_answer_rate | hard__accurate__neutral | 10 | 10 | 0.000 | [0.000, 0.000] |
| `Qwen/Qwen2.5-7B-Instruct` | validity_malfunctioning_minus_accurate | m1 | easy|hostile | 1 | 1 | -1.188 | - |
| `Qwen/Qwen2.5-7B-Instruct` | validity_malfunctioning_minus_accurate | m1 | easy|neutral | 4 | 4 | -1.797 | [-4.016, 0.062] |
| `Qwen/Qwen2.5-7B-Instruct` | validity_malfunctioning_minus_accurate | m1 | hard|hostile | 3 | 3 | -10.917 | [-31.250, 0.000] |
| `Qwen/Qwen2.5-7B-Instruct` | validity_malfunctioning_minus_accurate | m1 | hard|neutral | 7 | 7 | -6.143 | [-16.107, -0.375] |
| `Qwen/Qwen2.5-7B-Instruct` | validity_malfunctioning_minus_accurate | m2 | easy|hostile | 9 | 9 | 0.056 | [0.000, 0.144] |
| `Qwen/Qwen2.5-7B-Instruct` | validity_malfunctioning_minus_accurate | m2 | easy|neutral | 9 | 9 | 0.011 | [0.000, 0.033] |
| `Qwen/Qwen2.5-7B-Instruct` | validity_malfunctioning_minus_accurate | m2 | hard|hostile | 5 | 5 | 0.040 | [0.000, 0.080] |
| `Qwen/Qwen2.5-7B-Instruct` | validity_malfunctioning_minus_accurate | m2 | hard|neutral | 5 | 5 | 0.020 | [0.000, 0.060] |
| `Qwen/Qwen2.5-7B-Instruct` | validity_malfunctioning_minus_accurate | accuracy | easy|hostile | 10 | 10 | 0.000 | [0.000, 0.000] |
| `Qwen/Qwen2.5-7B-Instruct` | validity_malfunctioning_minus_accurate | accuracy | easy|neutral | 10 | 10 | 0.000 | [0.000, 0.000] |
| `Qwen/Qwen2.5-7B-Instruct` | validity_malfunctioning_minus_accurate | accuracy | hard|hostile | 7 | 7 | -0.143 | [-0.429, 0.000] |
| `Qwen/Qwen2.5-7B-Instruct` | validity_malfunctioning_minus_accurate | accuracy | hard|neutral | 9 | 9 | -0.222 | [-0.556, 0.000] |
| `Qwen/Qwen2.5-7B-Instruct` | validity_malfunctioning_minus_accurate | non_answer_rate | easy|hostile | 10 | 10 | 0.000 | [0.000, 0.000] |
| `Qwen/Qwen2.5-7B-Instruct` | validity_malfunctioning_minus_accurate | non_answer_rate | easy|neutral | 10 | 10 | 0.000 | [0.000, 0.000] |
| `Qwen/Qwen2.5-7B-Instruct` | validity_malfunctioning_minus_accurate | non_answer_rate | hard|hostile | 10 | 10 | -0.100 | [-0.300, 0.000] |
| `Qwen/Qwen2.5-7B-Instruct` | validity_malfunctioning_minus_accurate | non_answer_rate | hard|neutral | 10 | 10 | 0.000 | [0.000, 0.000] |
| `Qwen/Qwen2.5-7B-Instruct` | tone_hostile_minus_neutral | m1 | easy|accurate | 1 | 1 | -1.000 | - |
| `Qwen/Qwen2.5-7B-Instruct` | tone_hostile_minus_neutral | m1 | easy|malfunctioning_always_fail | 4 | 4 | 0.484 | [-0.703, 1.562] |
| `Qwen/Qwen2.5-7B-Instruct` | tone_hostile_minus_neutral | m1 | hard|accurate | 3 | 3 | -0.354 | [-0.938, 0.500] |
| `Qwen/Qwen2.5-7B-Instruct` | tone_hostile_minus_neutral | m1 | hard|malfunctioning_always_fail | 4 | 4 | 0.953 | [-0.719, 2.625] |
| `Qwen/Qwen2.5-7B-Instruct` | tone_hostile_minus_neutral | m2 | easy|accurate | 9 | 9 | -0.011 | [-0.033, 0.000] |
| `Qwen/Qwen2.5-7B-Instruct` | tone_hostile_minus_neutral | m2 | easy|malfunctioning_always_fail | 9 | 9 | 0.033 | [0.000, 0.100] |
| `Qwen/Qwen2.5-7B-Instruct` | tone_hostile_minus_neutral | m2 | hard|accurate | 5 | 5 | -0.040 | [-0.120, 0.000] |
| `Qwen/Qwen2.5-7B-Instruct` | tone_hostile_minus_neutral | m2 | hard|malfunctioning_always_fail | 5 | 5 | -0.020 | [-0.120, 0.060] |
| `Qwen/Qwen2.5-7B-Instruct` | tone_hostile_minus_neutral | accuracy | easy|accurate | 10 | 10 | 0.000 | [0.000, 0.000] |
| `Qwen/Qwen2.5-7B-Instruct` | tone_hostile_minus_neutral | accuracy | easy|malfunctioning_always_fail | 10 | 10 | 0.000 | [0.000, 0.000] |
| `Qwen/Qwen2.5-7B-Instruct` | tone_hostile_minus_neutral | accuracy | hard|accurate | 6 | 6 | 0.000 | [0.000, 0.000] |
| `Qwen/Qwen2.5-7B-Instruct` | tone_hostile_minus_neutral | accuracy | hard|malfunctioning_always_fail | 7 | 7 | 0.000 | [0.000, 0.000] |
| `Qwen/Qwen2.5-7B-Instruct` | tone_hostile_minus_neutral | non_answer_rate | easy|accurate | 10 | 10 | 0.000 | [0.000, 0.000] |
| `Qwen/Qwen2.5-7B-Instruct` | tone_hostile_minus_neutral | non_answer_rate | easy|malfunctioning_always_fail | 10 | 10 | 0.000 | [0.000, 0.000] |
| `Qwen/Qwen2.5-7B-Instruct` | tone_hostile_minus_neutral | non_answer_rate | hard|accurate | 10 | 10 | 0.200 | [-0.200, 0.600] |
| `Qwen/Qwen2.5-7B-Instruct` | tone_hostile_minus_neutral | non_answer_rate | hard|malfunctioning_always_fail | 10 | 10 | 0.100 | [-0.200, 0.400] |
| `Qwen/Qwen2.5-7B-Instruct` | recovery_minus_measured | m1 | easy__malfunctioning_always_fail__hostile | 5 | 5 | -4.588 | [-9.078, -1.600] |
| `Qwen/Qwen2.5-7B-Instruct` | recovery_minus_measured | m1 | easy__malfunctioning_always_fail__neutral | 5 | 5 | -3.850 | [-6.375, -1.625] |
| `Qwen/Qwen2.5-7B-Instruct` | recovery_minus_measured | m1 | hard__malfunctioning_always_fail__hostile | 5 | 5 | -3.225 | [-9.600, 3.150] |
| `Qwen/Qwen2.5-7B-Instruct` | recovery_minus_measured | m1 | hard__malfunctioning_always_fail__neutral | 7 | 7 | -5.170 | [-14.867, 2.054] |
| `Qwen/Qwen2.5-7B-Instruct` | recovery_minus_measured | m2 | easy__malfunctioning_always_fail__hostile | 8 | 8 | 0.000 | [0.000, 0.000] |
| `Qwen/Qwen2.5-7B-Instruct` | recovery_minus_measured | m2 | easy__malfunctioning_always_fail__neutral | 9 | 9 | 0.022 | [0.000, 0.067] |
| `Qwen/Qwen2.5-7B-Instruct` | recovery_minus_measured | m2 | hard__malfunctioning_always_fail__hostile | 4 | 4 | -0.025 | [-0.225, 0.150] |
| `Qwen/Qwen2.5-7B-Instruct` | recovery_minus_measured | m2 | hard__malfunctioning_always_fail__neutral | 5 | 5 | 0.100 | [0.020, 0.200] |
| `Qwen/Qwen2.5-7B-Instruct` | recovery_minus_measured | accuracy | easy__malfunctioning_always_fail__hostile | 10 | 10 | -0.100 | [-0.300, 0.000] |
| `Qwen/Qwen2.5-7B-Instruct` | recovery_minus_measured | accuracy | easy__malfunctioning_always_fail__neutral | 10 | 10 | 0.000 | [0.000, 0.000] |
| `Qwen/Qwen2.5-7B-Instruct` | recovery_minus_measured | accuracy | hard__malfunctioning_always_fail__hostile | 7 | 7 | 0.000 | [0.000, 0.000] |
| `Qwen/Qwen2.5-7B-Instruct` | recovery_minus_measured | accuracy | hard__malfunctioning_always_fail__neutral | 9 | 9 | -0.111 | [-0.333, 0.000] |
| `Qwen/Qwen2.5-7B-Instruct` | recovery_minus_measured | non_answer_rate | easy__malfunctioning_always_fail__hostile | 10 | 10 | 0.000 | [0.000, 0.000] |
| `Qwen/Qwen2.5-7B-Instruct` | recovery_minus_measured | non_answer_rate | easy__malfunctioning_always_fail__neutral | 10 | 10 | 0.000 | [0.000, 0.000] |
| `Qwen/Qwen2.5-7B-Instruct` | recovery_minus_measured | non_answer_rate | hard__malfunctioning_always_fail__hostile | 10 | 10 | 0.000 | [-0.300, 0.300] |
| `Qwen/Qwen2.5-7B-Instruct` | recovery_minus_measured | non_answer_rate | hard__malfunctioning_always_fail__neutral | 10 | 10 | 0.000 | [0.000, 0.000] |
| `Qwen/Qwen2.5-7B-Instruct` | onset_minus_measured | m1 | easy__accurate__hostile | 2 | 2 | -5.156 | [-7.750, -2.562] |
| `Qwen/Qwen2.5-7B-Instruct` | onset_minus_measured | m1 | easy__accurate__neutral | 5 | 5 | -5.762 | [-9.087, -2.438] |
| `Qwen/Qwen2.5-7B-Instruct` | onset_minus_measured | m1 | hard__accurate__hostile | 2 | 2 | -4.125 | [-5.625, -2.625] |
| `Qwen/Qwen2.5-7B-Instruct` | onset_minus_measured | m1 | hard__accurate__neutral | 8 | 8 | -3.320 | [-5.039, -1.109] |
| `Qwen/Qwen2.5-7B-Instruct` | onset_minus_measured | m2 | easy__accurate__hostile | 10 | 10 | 0.000 | [0.000, 0.000] |
| `Qwen/Qwen2.5-7B-Instruct` | onset_minus_measured | m2 | easy__accurate__neutral | 9 | 9 | 0.000 | [0.000, 0.000] |
| `Qwen/Qwen2.5-7B-Instruct` | onset_minus_measured | m2 | hard__accurate__hostile | 6 | 6 | 0.033 | [-0.050, 0.150] |
| `Qwen/Qwen2.5-7B-Instruct` | onset_minus_measured | m2 | hard__accurate__neutral | 4 | 4 | 0.025 | [0.000, 0.075] |
| `Qwen/Qwen2.5-7B-Instruct` | onset_minus_measured | accuracy | easy__accurate__hostile | 10 | 10 | 0.000 | [0.000, 0.000] |
| `Qwen/Qwen2.5-7B-Instruct` | onset_minus_measured | accuracy | easy__accurate__neutral | 10 | 10 | 0.000 | [0.000, 0.000] |
| `Qwen/Qwen2.5-7B-Instruct` | onset_minus_measured | accuracy | hard__accurate__hostile | 6 | 6 | 0.000 | [0.000, 0.000] |
| `Qwen/Qwen2.5-7B-Instruct` | onset_minus_measured | accuracy | hard__accurate__neutral | 9 | 9 | 0.000 | [0.000, 0.000] |
| `Qwen/Qwen2.5-7B-Instruct` | onset_minus_measured | non_answer_rate | easy__accurate__hostile | 10 | 10 | 0.000 | [0.000, 0.000] |
| `Qwen/Qwen2.5-7B-Instruct` | onset_minus_measured | non_answer_rate | easy__accurate__neutral | 10 | 10 | 0.000 | [0.000, 0.000] |
| `Qwen/Qwen2.5-7B-Instruct` | onset_minus_measured | non_answer_rate | hard__accurate__hostile | 10 | 10 | 0.100 | [0.000, 0.300] |
| `Qwen/Qwen2.5-7B-Instruct` | onset_minus_measured | non_answer_rate | hard__accurate__neutral | 10 | 10 | 0.000 | [0.000, 0.000] |
| `Qwen/Qwen2.5-7B-Instruct` | washout_minus_onset | m1 | easy__accurate__hostile | 5 | 5 | 4.088 | [2.900, 5.228] |
| `Qwen/Qwen2.5-7B-Instruct` | washout_minus_onset | m1 | easy__accurate__neutral | 10 | 10 | 4.262 | [2.100, 6.738] |
| `Qwen/Qwen2.5-7B-Instruct` | washout_minus_onset | m1 | hard__accurate__hostile | 4 | 4 | 1.625 | [-0.188, 3.875] |
| `Qwen/Qwen2.5-7B-Instruct` | washout_minus_onset | m1 | hard__accurate__neutral | 8 | 8 | 3.008 | [1.664, 4.470] |
| `Qwen/Qwen2.5-7B-Instruct` | washout_minus_onset | m2 | easy__accurate__hostile | 10 | 10 | 0.000 | [0.000, 0.000] |
| `Qwen/Qwen2.5-7B-Instruct` | washout_minus_onset | m2 | easy__accurate__neutral | 9 | 9 | -0.011 | [-0.033, 0.000] |
| `Qwen/Qwen2.5-7B-Instruct` | washout_minus_onset | m2 | hard__accurate__hostile | 5 | 5 | 0.000 | [0.000, 0.000] |
| `Qwen/Qwen2.5-7B-Instruct` | washout_minus_onset | m2 | hard__accurate__neutral | 4 | 4 | -0.025 | [-0.075, 0.000] |
| `Qwen/Qwen2.5-7B-Instruct` | washout_minus_onset | accuracy | easy__accurate__hostile | 10 | 10 | 0.000 | [0.000, 0.000] |
| `Qwen/Qwen2.5-7B-Instruct` | washout_minus_onset | accuracy | easy__accurate__neutral | 10 | 10 | 0.000 | [0.000, 0.000] |
| `Qwen/Qwen2.5-7B-Instruct` | washout_minus_onset | accuracy | hard__accurate__hostile | 6 | 6 | 0.000 | [0.000, 0.000] |
| `Qwen/Qwen2.5-7B-Instruct` | washout_minus_onset | accuracy | hard__accurate__neutral | 9 | 9 | 0.000 | [0.000, 0.000] |
| `Qwen/Qwen2.5-7B-Instruct` | washout_minus_onset | non_answer_rate | easy__accurate__hostile | 10 | 10 | 0.000 | [0.000, 0.000] |
| `Qwen/Qwen2.5-7B-Instruct` | washout_minus_onset | non_answer_rate | easy__accurate__neutral | 10 | 10 | 0.000 | [0.000, 0.000] |
| `Qwen/Qwen2.5-7B-Instruct` | washout_minus_onset | non_answer_rate | hard__accurate__hostile | 10 | 10 | -0.100 | [-0.300, 0.000] |
| `Qwen/Qwen2.5-7B-Instruct` | washout_minus_onset | non_answer_rate | hard__accurate__neutral | 10 | 10 | 0.000 | [0.000, 0.000] |
| `google/gemma-2-2b-it` | validity_malfunctioning_minus_accurate | m1 | easy|hostile | 10 | 10 | 0.116 | [-0.494, 0.791] |
| `google/gemma-2-2b-it` | validity_malfunctioning_minus_accurate | m1 | easy|neutral | 10 | 10 | -0.706 | [-1.119, -0.313] |
| `google/gemma-2-2b-it` | validity_malfunctioning_minus_accurate | m1 | hard|hostile | 8 | 8 | -0.473 | [-7.348, 5.480] |
| `google/gemma-2-2b-it` | validity_malfunctioning_minus_accurate | m1 | hard|neutral | 8 | 8 | -3.891 | [-9.633, 0.078] |
| `google/gemma-2-2b-it` | validity_malfunctioning_minus_accurate | m2 | easy|hostile | 8 | 8 | -0.013 | [-0.100, 0.050] |
| `google/gemma-2-2b-it` | validity_malfunctioning_minus_accurate | m2 | easy|neutral | 9 | 9 | 0.089 | [0.011, 0.178] |
| `google/gemma-2-2b-it` | validity_malfunctioning_minus_accurate | m2 | hard|hostile | 3 | 3 | 0.133 | [0.000, 0.300] |
| `google/gemma-2-2b-it` | validity_malfunctioning_minus_accurate | m2 | hard|neutral | 2 | 2 | 0.050 | [0.000, 0.100] |
| `google/gemma-2-2b-it` | validity_malfunctioning_minus_accurate | accuracy | easy|hostile | 10 | 10 | 0.000 | [0.000, 0.000] |
| `google/gemma-2-2b-it` | validity_malfunctioning_minus_accurate | accuracy | easy|neutral | 10 | 10 | 0.000 | [0.000, 0.000] |
| `google/gemma-2-2b-it` | validity_malfunctioning_minus_accurate | accuracy | hard|hostile | 8 | 8 | 0.000 | [-0.375, 0.375] |
| `google/gemma-2-2b-it` | validity_malfunctioning_minus_accurate | accuracy | hard|neutral | 8 | 8 | -0.250 | [-0.625, 0.000] |
| `google/gemma-2-2b-it` | validity_malfunctioning_minus_accurate | non_answer_rate | easy|hostile | 10 | 10 | 0.000 | [0.000, 0.000] |
| `google/gemma-2-2b-it` | validity_malfunctioning_minus_accurate | non_answer_rate | easy|neutral | 10 | 10 | 0.000 | [0.000, 0.000] |
| `google/gemma-2-2b-it` | validity_malfunctioning_minus_accurate | non_answer_rate | hard|hostile | 10 | 10 | -0.100 | [-0.300, 0.000] |
| `google/gemma-2-2b-it` | validity_malfunctioning_minus_accurate | non_answer_rate | hard|neutral | 10 | 10 | -0.100 | [-0.300, 0.000] |
| `google/gemma-2-2b-it` | tone_hostile_minus_neutral | m1 | easy|accurate | 10 | 10 | 1.241 | [-1.156, 5.103] |
| `google/gemma-2-2b-it` | tone_hostile_minus_neutral | m1 | easy|malfunctioning_always_fail | 10 | 10 | 2.063 | [-0.013, 5.663] |
| `google/gemma-2-2b-it` | tone_hostile_minus_neutral | m1 | hard|accurate | 7 | 7 | -4.049 | [-10.350, 1.250] |
| `google/gemma-2-2b-it` | tone_hostile_minus_neutral | m1 | hard|malfunctioning_always_fail | 8 | 8 | -1.547 | [-5.875, 1.305] |
| `google/gemma-2-2b-it` | tone_hostile_minus_neutral | m2 | easy|accurate | 9 | 9 | 0.056 | [0.000, 0.144] |
| `google/gemma-2-2b-it` | tone_hostile_minus_neutral | m2 | easy|malfunctioning_always_fail | 7 | 7 | -0.057 | [-0.143, 0.000] |
| `google/gemma-2-2b-it` | tone_hostile_minus_neutral | m2 | hard|accurate | 3 | 3 | -0.100 | [-0.200, 0.000] |
| `google/gemma-2-2b-it` | tone_hostile_minus_neutral | m2 | hard|malfunctioning_always_fail | 1 | 1 | 0.000 | - |
| `google/gemma-2-2b-it` | tone_hostile_minus_neutral | accuracy | easy|accurate | 10 | 10 | 0.100 | [0.000, 0.300] |
| `google/gemma-2-2b-it` | tone_hostile_minus_neutral | accuracy | easy|malfunctioning_always_fail | 10 | 10 | 0.100 | [0.000, 0.300] |
| `google/gemma-2-2b-it` | tone_hostile_minus_neutral | accuracy | hard|accurate | 7 | 7 | -0.286 | [-0.571, 0.000] |
| `google/gemma-2-2b-it` | tone_hostile_minus_neutral | accuracy | hard|malfunctioning_always_fail | 8 | 8 | -0.125 | [-0.375, 0.000] |
| `google/gemma-2-2b-it` | tone_hostile_minus_neutral | non_answer_rate | easy|accurate | 10 | 10 | 0.000 | [0.000, 0.000] |
| `google/gemma-2-2b-it` | tone_hostile_minus_neutral | non_answer_rate | easy|malfunctioning_always_fail | 10 | 10 | 0.000 | [0.000, 0.000] |
| `google/gemma-2-2b-it` | tone_hostile_minus_neutral | non_answer_rate | hard|accurate | 10 | 10 | 0.000 | [-0.300, 0.300] |
| `google/gemma-2-2b-it` | tone_hostile_minus_neutral | non_answer_rate | hard|malfunctioning_always_fail | 10 | 10 | 0.000 | [-0.300, 0.300] |
| `google/gemma-2-2b-it` | recovery_minus_measured | m1 | easy__malfunctioning_always_fail__hostile | 10 | 10 | -0.103 | [-0.397, 0.209] |
| `google/gemma-2-2b-it` | recovery_minus_measured | m1 | easy__malfunctioning_always_fail__neutral | 9 | 9 | -1.663 | [-5.462, 0.459] |
| `google/gemma-2-2b-it` | recovery_minus_measured | m1 | hard__malfunctioning_always_fail__hostile | 8 | 8 | -0.664 | [-1.938, 0.336] |
| `google/gemma-2-2b-it` | recovery_minus_measured | m1 | hard__malfunctioning_always_fail__neutral | 8 | 8 | 0.281 | [-0.539, 1.164] |
| `google/gemma-2-2b-it` | recovery_minus_measured | m2 | easy__malfunctioning_always_fail__hostile | 5 | 5 | 0.000 | [0.000, 0.000] |
| `google/gemma-2-2b-it` | recovery_minus_measured | m2 | easy__malfunctioning_always_fail__neutral | 7 | 7 | 0.000 | [-0.114, 0.086] |
| `google/gemma-2-2b-it` | recovery_minus_measured | m2 | hard__malfunctioning_always_fail__hostile | 2 | 2 | -0.050 | [-0.100, 0.000] |
| `google/gemma-2-2b-it` | recovery_minus_measured | m2 | hard__malfunctioning_always_fail__neutral | 2 | 2 | 0.150 | [0.000, 0.300] |
| `google/gemma-2-2b-it` | recovery_minus_measured | accuracy | easy__malfunctioning_always_fail__hostile | 10 | 10 | 0.000 | [0.000, 0.000] |
| `google/gemma-2-2b-it` | recovery_minus_measured | accuracy | easy__malfunctioning_always_fail__neutral | 9 | 9 | -0.111 | [-0.333, 0.000] |
| `google/gemma-2-2b-it` | recovery_minus_measured | accuracy | hard__malfunctioning_always_fail__hostile | 9 | 9 | 0.000 | [0.000, 0.000] |
| `google/gemma-2-2b-it` | recovery_minus_measured | accuracy | hard__malfunctioning_always_fail__neutral | 8 | 8 | 0.000 | [0.000, 0.000] |
| `google/gemma-2-2b-it` | recovery_minus_measured | non_answer_rate | easy__malfunctioning_always_fail__hostile | 10 | 10 | 0.000 | [0.000, 0.000] |
| `google/gemma-2-2b-it` | recovery_minus_measured | non_answer_rate | easy__malfunctioning_always_fail__neutral | 10 | 10 | 0.100 | [0.000, 0.300] |
| `google/gemma-2-2b-it` | recovery_minus_measured | non_answer_rate | hard__malfunctioning_always_fail__hostile | 10 | 10 | 0.000 | [0.000, 0.000] |
| `google/gemma-2-2b-it` | recovery_minus_measured | non_answer_rate | hard__malfunctioning_always_fail__neutral | 10 | 10 | 0.100 | [0.000, 0.300] |
| `google/gemma-2-2b-it` | onset_minus_measured | m1 | easy__accurate__hostile | 6 | 6 | -2.010 | [-5.104, 0.042] |
| `google/gemma-2-2b-it` | onset_minus_measured | m1 | easy__accurate__neutral | 10 | 10 | -1.234 | [-2.841, -0.094] |
| `google/gemma-2-2b-it` | onset_minus_measured | m1 | hard__accurate__hostile | 2 | 2 | 0.187 | [-0.500, 0.875] |
| `google/gemma-2-2b-it` | onset_minus_measured | m1 | hard__accurate__neutral | 5 | 5 | -2.875 | [-6.350, -0.550] |
| `google/gemma-2-2b-it` | onset_minus_measured | m2 | easy__accurate__neutral | 6 | 6 | 0.017 | [0.000, 0.050] |
| `google/gemma-2-2b-it` | onset_minus_measured | m2 | hard__accurate__neutral | 1 | 1 | 0.000 | - |
| `google/gemma-2-2b-it` | onset_minus_measured | accuracy | easy__accurate__hostile | 6 | 6 | -0.167 | [-0.500, 0.000] |
| `google/gemma-2-2b-it` | onset_minus_measured | accuracy | easy__accurate__neutral | 10 | 10 | 0.000 | [0.000, 0.000] |
| `google/gemma-2-2b-it` | onset_minus_measured | accuracy | hard__accurate__hostile | 2 | 2 | 0.000 | [0.000, 0.000] |
| `google/gemma-2-2b-it` | onset_minus_measured | accuracy | hard__accurate__neutral | 5 | 5 | -0.200 | [-0.600, 0.000] |
| `google/gemma-2-2b-it` | onset_minus_measured | non_answer_rate | easy__accurate__hostile | 10 | 10 | 0.400 | [0.100, 0.700] |
| `google/gemma-2-2b-it` | onset_minus_measured | non_answer_rate | easy__accurate__neutral | 10 | 10 | 0.000 | [0.000, 0.000] |
| `google/gemma-2-2b-it` | onset_minus_measured | non_answer_rate | hard__accurate__hostile | 10 | 10 | 0.600 | [0.300, 0.900] |
| `google/gemma-2-2b-it` | onset_minus_measured | non_answer_rate | hard__accurate__neutral | 10 | 10 | 0.300 | [0.000, 0.600] |
| `google/gemma-2-2b-it` | washout_minus_onset | m1 | easy__accurate__hostile | 6 | 6 | 1.859 | [0.656, 3.427] |
| `google/gemma-2-2b-it` | washout_minus_onset | m1 | easy__accurate__neutral | 10 | 10 | 0.866 | [0.231, 1.835] |
| `google/gemma-2-2b-it` | washout_minus_onset | m1 | hard__accurate__hostile | 2 | 2 | -0.469 | [-0.750, -0.187] |
| `google/gemma-2-2b-it` | washout_minus_onset | m1 | hard__accurate__neutral | 5 | 5 | 0.563 | [-1.312, 1.888] |
| `google/gemma-2-2b-it` | washout_minus_onset | m2 | easy__accurate__neutral | 6 | 6 | -0.033 | [-0.100, 0.000] |
| `google/gemma-2-2b-it` | washout_minus_onset | m2 | hard__accurate__neutral | 1 | 1 | 0.000 | - |
| `google/gemma-2-2b-it` | washout_minus_onset | accuracy | easy__accurate__hostile | 6 | 6 | 0.167 | [0.000, 0.500] |
| `google/gemma-2-2b-it` | washout_minus_onset | accuracy | easy__accurate__neutral | 10 | 10 | 0.000 | [0.000, 0.000] |
| `google/gemma-2-2b-it` | washout_minus_onset | accuracy | hard__accurate__hostile | 2 | 2 | 0.000 | [0.000, 0.000] |
| `google/gemma-2-2b-it` | washout_minus_onset | accuracy | hard__accurate__neutral | 5 | 5 | 0.000 | [0.000, 0.000] |
| `google/gemma-2-2b-it` | washout_minus_onset | non_answer_rate | easy__accurate__hostile | 10 | 10 | -0.400 | [-0.700, -0.100] |
| `google/gemma-2-2b-it` | washout_minus_onset | non_answer_rate | easy__accurate__neutral | 10 | 10 | 0.000 | [0.000, 0.000] |
| `google/gemma-2-2b-it` | washout_minus_onset | non_answer_rate | hard__accurate__hostile | 10 | 10 | -0.500 | [-0.800, -0.200] |
| `google/gemma-2-2b-it` | washout_minus_onset | non_answer_rate | hard__accurate__neutral | 10 | 10 | -0.100 | [-0.300, 0.000] |
| `google/gemma-2-9b-it` | validity_malfunctioning_minus_accurate | m1 | easy|hostile | 10 | 10 | -2.497 | [-8.188, 1.485] |
| `google/gemma-2-9b-it` | validity_malfunctioning_minus_accurate | m1 | easy|neutral | 10 | 10 | -3.800 | [-5.297, -2.350] |
| `google/gemma-2-9b-it` | validity_malfunctioning_minus_accurate | m1 | hard|hostile | 7 | 7 | 4.179 | [-5.545, 14.581] |
| `google/gemma-2-9b-it` | validity_malfunctioning_minus_accurate | m1 | hard|neutral | 8 | 8 | -1.230 | [-2.922, 0.481] |
| `google/gemma-2-9b-it` | validity_malfunctioning_minus_accurate | m2 | easy|hostile | 7 | 7 | -0.257 | [-0.400, -0.100] |
| `google/gemma-2-9b-it` | validity_malfunctioning_minus_accurate | m2 | easy|neutral | 10 | 10 | 0.000 | [0.000, 0.000] |
| `google/gemma-2-9b-it` | validity_malfunctioning_minus_accurate | m2 | hard|hostile | 5 | 5 | -0.040 | [-0.180, 0.140] |
| `google/gemma-2-9b-it` | validity_malfunctioning_minus_accurate | m2 | hard|neutral | 6 | 6 | 0.167 | [0.033, 0.333] |
| `google/gemma-2-9b-it` | validity_malfunctioning_minus_accurate | accuracy | easy|hostile | 10 | 10 | -0.100 | [-0.300, 0.000] |
| `google/gemma-2-9b-it` | validity_malfunctioning_minus_accurate | accuracy | easy|neutral | 10 | 10 | 0.000 | [0.000, 0.000] |
| `google/gemma-2-9b-it` | validity_malfunctioning_minus_accurate | accuracy | hard|hostile | 7 | 7 | 0.143 | [-0.286, 0.571] |
| `google/gemma-2-9b-it` | validity_malfunctioning_minus_accurate | accuracy | hard|neutral | 8 | 8 | 0.000 | [0.000, 0.000] |
| `google/gemma-2-9b-it` | validity_malfunctioning_minus_accurate | non_answer_rate | easy|hostile | 10 | 10 | 0.000 | [0.000, 0.000] |
| `google/gemma-2-9b-it` | validity_malfunctioning_minus_accurate | non_answer_rate | easy|neutral | 10 | 10 | 0.000 | [0.000, 0.000] |
| `google/gemma-2-9b-it` | validity_malfunctioning_minus_accurate | non_answer_rate | hard|hostile | 10 | 10 | -0.100 | [-0.300, 0.000] |
| `google/gemma-2-9b-it` | validity_malfunctioning_minus_accurate | non_answer_rate | hard|neutral | 10 | 10 | 0.000 | [0.000, 0.000] |
| `google/gemma-2-9b-it` | tone_hostile_minus_neutral | m1 | easy|accurate | 10 | 10 | -2.275 | [-3.903, -1.000] |
| `google/gemma-2-9b-it` | tone_hostile_minus_neutral | m1 | easy|malfunctioning_always_fail | 10 | 10 | -0.972 | [-6.616, 2.675] |
| `google/gemma-2-9b-it` | tone_hostile_minus_neutral | m1 | hard|accurate | 7 | 7 | -8.781 | [-17.277, -1.268] |
| `google/gemma-2-9b-it` | tone_hostile_minus_neutral | m1 | hard|malfunctioning_always_fail | 8 | 8 | -2.047 | [-8.594, 1.969] |
| `google/gemma-2-9b-it` | tone_hostile_minus_neutral | m2 | easy|accurate | 7 | 7 | 0.257 | [0.100, 0.386] |
| `google/gemma-2-9b-it` | tone_hostile_minus_neutral | m2 | easy|malfunctioning_always_fail | 8 | 8 | 0.000 | [0.000, 0.000] |
| `google/gemma-2-9b-it` | tone_hostile_minus_neutral | m2 | hard|accurate | 6 | 6 | 0.183 | [0.100, 0.267] |
| `google/gemma-2-9b-it` | tone_hostile_minus_neutral | m2 | hard|malfunctioning_always_fail | 5 | 5 | -0.080 | [-0.220, 0.040] |
| `google/gemma-2-9b-it` | tone_hostile_minus_neutral | accuracy | easy|accurate | 10 | 10 | 0.000 | [0.000, 0.000] |
| `google/gemma-2-9b-it` | tone_hostile_minus_neutral | accuracy | easy|malfunctioning_always_fail | 10 | 10 | -0.100 | [-0.300, 0.000] |
| `google/gemma-2-9b-it` | tone_hostile_minus_neutral | accuracy | hard|accurate | 7 | 7 | -0.286 | [-0.571, 0.000] |
| `google/gemma-2-9b-it` | tone_hostile_minus_neutral | accuracy | hard|malfunctioning_always_fail | 8 | 8 | -0.125 | [-0.375, 0.000] |
| `google/gemma-2-9b-it` | tone_hostile_minus_neutral | non_answer_rate | easy|accurate | 10 | 10 | 0.000 | [0.000, 0.000] |
| `google/gemma-2-9b-it` | tone_hostile_minus_neutral | non_answer_rate | easy|malfunctioning_always_fail | 10 | 10 | 0.000 | [0.000, 0.000] |
| `google/gemma-2-9b-it` | tone_hostile_minus_neutral | non_answer_rate | hard|accurate | 10 | 10 | 0.100 | [0.000, 0.300] |
| `google/gemma-2-9b-it` | tone_hostile_minus_neutral | non_answer_rate | hard|malfunctioning_always_fail | 10 | 10 | 0.000 | [0.000, 0.000] |
| `google/gemma-2-9b-it` | recovery_minus_measured | m1 | easy__malfunctioning_always_fail__hostile | 10 | 10 | 0.941 | [-2.125, 6.091] |
| `google/gemma-2-9b-it` | recovery_minus_measured | m1 | easy__malfunctioning_always_fail__neutral | 10 | 10 | -1.363 | [-3.031, 0.294] |
| `google/gemma-2-9b-it` | recovery_minus_measured | m1 | hard__malfunctioning_always_fail__hostile | 7 | 7 | -1.205 | [-1.777, -0.536] |
| `google/gemma-2-9b-it` | recovery_minus_measured | m1 | hard__malfunctioning_always_fail__neutral | 8 | 8 | -5.797 | [-11.383, -1.133] |
| `google/gemma-2-9b-it` | recovery_minus_measured | m2 | easy__malfunctioning_always_fail__hostile | 8 | 8 | -0.012 | [-0.037, 0.000] |
| `google/gemma-2-9b-it` | recovery_minus_measured | m2 | easy__malfunctioning_always_fail__neutral | 10 | 10 | -0.010 | [-0.030, 0.000] |
| `google/gemma-2-9b-it` | recovery_minus_measured | m2 | hard__malfunctioning_always_fail__hostile | 3 | 3 | -0.067 | [-0.200, 0.000] |
| `google/gemma-2-9b-it` | recovery_minus_measured | m2 | hard__malfunctioning_always_fail__neutral | 6 | 6 | 0.000 | [-0.100, 0.100] |
| `google/gemma-2-9b-it` | recovery_minus_measured | accuracy | easy__malfunctioning_always_fail__hostile | 10 | 10 | 0.100 | [0.000, 0.300] |
| `google/gemma-2-9b-it` | recovery_minus_measured | accuracy | easy__malfunctioning_always_fail__neutral | 10 | 10 | 0.000 | [0.000, 0.000] |
| `google/gemma-2-9b-it` | recovery_minus_measured | accuracy | hard__malfunctioning_always_fail__hostile | 7 | 7 | 0.000 | [0.000, 0.000] |
| `google/gemma-2-9b-it` | recovery_minus_measured | accuracy | hard__malfunctioning_always_fail__neutral | 8 | 8 | -0.250 | [-0.625, 0.000] |
| `google/gemma-2-9b-it` | recovery_minus_measured | non_answer_rate | easy__malfunctioning_always_fail__hostile | 10 | 10 | 0.000 | [0.000, 0.000] |
| `google/gemma-2-9b-it` | recovery_minus_measured | non_answer_rate | easy__malfunctioning_always_fail__neutral | 10 | 10 | 0.000 | [0.000, 0.000] |
| `google/gemma-2-9b-it` | recovery_minus_measured | non_answer_rate | hard__malfunctioning_always_fail__hostile | 10 | 10 | 0.100 | [0.000, 0.300] |
| `google/gemma-2-9b-it` | recovery_minus_measured | non_answer_rate | hard__malfunctioning_always_fail__neutral | 10 | 10 | 0.000 | [0.000, 0.000] |
| `google/gemma-2-9b-it` | onset_minus_measured | m1 | easy__accurate__hostile | 9 | 9 | -6.181 | [-10.250, -2.250] |
| `google/gemma-2-9b-it` | onset_minus_measured | m1 | easy__accurate__neutral | 10 | 10 | -3.459 | [-4.450, -2.612] |
| `google/gemma-2-9b-it` | onset_minus_measured | m1 | hard__accurate__hostile | 5 | 5 | 1.125 | [-0.877, 3.225] |
| `google/gemma-2-9b-it` | onset_minus_measured | m1 | hard__accurate__neutral | 8 | 8 | -1.723 | [-6.023, 2.059] |
| `google/gemma-2-9b-it` | onset_minus_measured | m2 | easy__accurate__hostile | 1 | 1 | 0.000 | - |
| `google/gemma-2-9b-it` | onset_minus_measured | m2 | easy__accurate__neutral | 10 | 10 | 0.000 | [0.000, 0.000] |
| `google/gemma-2-9b-it` | onset_minus_measured | m2 | hard__accurate__neutral | 6 | 6 | 0.017 | [0.000, 0.050] |
| `google/gemma-2-9b-it` | onset_minus_measured | accuracy | easy__accurate__hostile | 9 | 9 | -0.222 | [-0.556, 0.000] |
| `google/gemma-2-9b-it` | onset_minus_measured | accuracy | easy__accurate__neutral | 10 | 10 | 0.000 | [0.000, 0.000] |
| `google/gemma-2-9b-it` | onset_minus_measured | accuracy | hard__accurate__hostile | 5 | 5 | 0.000 | [0.000, 0.000] |
| `google/gemma-2-9b-it` | onset_minus_measured | accuracy | hard__accurate__neutral | 8 | 8 | -0.125 | [-0.375, 0.000] |
| `google/gemma-2-9b-it` | onset_minus_measured | non_answer_rate | easy__accurate__hostile | 10 | 10 | 0.100 | [0.000, 0.300] |
| `google/gemma-2-9b-it` | onset_minus_measured | non_answer_rate | easy__accurate__neutral | 10 | 10 | 0.000 | [0.000, 0.000] |
| `google/gemma-2-9b-it` | onset_minus_measured | non_answer_rate | hard__accurate__hostile | 10 | 10 | 0.200 | [0.000, 0.500] |
| `google/gemma-2-9b-it` | onset_minus_measured | non_answer_rate | hard__accurate__neutral | 10 | 10 | 0.000 | [0.000, 0.000] |
| `google/gemma-2-9b-it` | washout_minus_onset | m1 | easy__accurate__hostile | 9 | 9 | 4.726 | [0.302, 10.523] |
| `google/gemma-2-9b-it` | washout_minus_onset | m1 | easy__accurate__neutral | 10 | 10 | 1.737 | [0.947, 2.441] |
| `google/gemma-2-9b-it` | washout_minus_onset | m1 | hard__accurate__hostile | 5 | 5 | -0.237 | [-3.537, 3.063] |
| `google/gemma-2-9b-it` | washout_minus_onset | m1 | hard__accurate__neutral | 8 | 8 | -0.383 | [-2.758, 1.789] |
| `google/gemma-2-9b-it` | washout_minus_onset | m2 | easy__accurate__hostile | 1 | 1 | 0.000 | - |
| `google/gemma-2-9b-it` | washout_minus_onset | m2 | easy__accurate__neutral | 10 | 10 | 0.000 | [0.000, 0.000] |
| `google/gemma-2-9b-it` | washout_minus_onset | m2 | hard__accurate__neutral | 6 | 6 | 0.000 | [0.000, 0.000] |
| `google/gemma-2-9b-it` | washout_minus_onset | accuracy | easy__accurate__hostile | 9 | 9 | 0.111 | [0.000, 0.333] |
| `google/gemma-2-9b-it` | washout_minus_onset | accuracy | easy__accurate__neutral | 10 | 10 | 0.000 | [0.000, 0.000] |
| `google/gemma-2-9b-it` | washout_minus_onset | accuracy | hard__accurate__hostile | 5 | 5 | 0.000 | [0.000, 0.000] |
| `google/gemma-2-9b-it` | washout_minus_onset | accuracy | hard__accurate__neutral | 8 | 8 | 0.000 | [0.000, 0.000] |
| `google/gemma-2-9b-it` | washout_minus_onset | non_answer_rate | easy__accurate__hostile | 10 | 10 | -0.100 | [-0.300, 0.000] |
| `google/gemma-2-9b-it` | washout_minus_onset | non_answer_rate | easy__accurate__neutral | 10 | 10 | 0.000 | [0.000, 0.000] |
| `google/gemma-2-9b-it` | washout_minus_onset | non_answer_rate | hard__accurate__hostile | 10 | 10 | -0.200 | [-0.500, 0.000] |
| `google/gemma-2-9b-it` | washout_minus_onset | non_answer_rate | hard__accurate__neutral | 10 | 10 | 0.000 | [0.000, 0.000] |
