| source | model | mse | mae | latency_p95_ms | capacity_cost | params | max_memory_mb |
| --- | --- | --- | --- | --- | --- | --- | --- |
| alibaba2018 | race_dlinear | 0.2671 | 0.3783 | 0.355 | 0.4107 | 4668 | 17 |
| alibaba2018 | dlinear | 0.2697 | 0.3837 | 0.1369 | 0.4056 | 2334 | 17 |
| alibaba2018 | official_itransformer_native | 0.2744 | 0.3744 | 1.96 | 0.3814 | 2.807e+05 | 30 |
| alibaba2018 | official_patchtst | 0.2748 | 0.3764 | 2.15 | 0.3814 | 3.056e+05 | 155 |
| alibaba2018 | patchtst | 0.2753 | 0.3845 | 0.6482 | 0.4246 | 1.242e+05 | 37 |
| alibaba2018 | official_timemixer | 0.277 | 0.3799 | 4.969 | 0.3826 | 1.288e+05 | 178 |
| alibaba2018 | chronos_bolt_reference | 0.3922 | 0.4705 | 10.95 | 0.3942 |  | 26 |
| alibaba2018 | last_value | 0.4082 | 0.4475 | 0.07342 | 0.4657 | 0 | 0 |
| gaia | last_value | 8.783e-04 | 0.003656 | 0.07375 | 289.2 | 0 | 0 |
| gaia | dlinear | 9.428e-04 | 0.008057 | 0.1096 | 6.719e+08 | 2330 | 16 |
| gaia | race_dlinear | 0.001026 | 0.01064 | 0.2926 | 3.132e+08 | 4660 | 16 |
| gaia | patchtst | 0.02429 | 0.04014 | 0.6527 | 3.089e+09 | 1.201e+05 | 36 |
| netman | patchtst | 0.00473 | 0.03666 | 0.9261 | 5.823 | 1.201e+05 | 36 |
| netman | dlinear | 0.004915 | 0.0318 | 0.2051 | 1.961 | 2330 | 16 |
| netman | race_dlinear | 0.005115 | 0.03255 | 0.7466 | 2.696 | 4660 | 16 |
| netman | last_value | 0.005397 | 0.03188 | 0.07803 | 1.352 | 0 | 0 |
| salesforce_borg | chronos_bolt_reference | 4.720e-06 | 0.001965 | 10.85 | 0.01333 |  | 26 |
| salesforce_borg | official_timemixer | 0.02475 | 0.078 | 3.678 | 0.1285 | 1.294e+05 | 179 |
| salesforce_borg | official_patchtst | 0.02533 | 0.08147 | 1.588 | 0.1335 | 3.056e+05 | 209 |
| salesforce_borg | official_itransformer_native | 0.02804 | 0.08466 | 1.723 | 0.1315 | 2.807e+05 | 34 |
