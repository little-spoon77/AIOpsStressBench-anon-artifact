| source | dataset | stress | latency_budget_ms | selected_model | route_reason | mse | mae | capacity_cost | latency_p95_ms | mse_vs_dlinear | capacity_cost_vs_dlinear | latency_vs_dlinear |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| alibaba2018 | alibaba2018 | burst | 0.2 | dlinear | strict_latency_rule | 0.4432 | 0.5351 | 0.6318 | 0.1422 | -0.001399 | -8.679e-06 | 0 |
| alibaba2018 | alibaba2018 | burst | 0.5 | race_dlinear | validation_best_capacity_cost | 0.4264 | 0.5247 | 0.6238 | 0.3623 | -0.03929 | -0.01261 | 1.548 |
| alibaba2018 | alibaba2018 | burst | 1 | patchtst | validation_best_capacity_cost | 0.3232 | 0.4191 | 0.4485 | 0.9882 | -0.2717 | -0.2901 | 5.949 |
| alibaba2018 | alibaba2018 | burst |  | patchtst | validation_best_capacity_cost | 0.3232 | 0.4191 | 0.4485 | 0.9882 | -0.2717 | -0.2901 | 5.949 |
| alibaba2018 | alibaba2018 | clean | 0.2 | dlinear | strict_latency_rule | 0.4091 | 0.5098 | 0.6292 | 0.1422 | 0 | 0 | 0 |
| alibaba2018 | alibaba2018 | clean | 0.5 | race_dlinear | validation_best_capacity_cost | 0.3923 | 0.4995 | 0.622 | 0.3623 | -0.04102 | -0.01142 | 1.548 |
| alibaba2018 | alibaba2018 | clean | 1 | patchtst | validation_best_capacity_cost | 0.3128 | 0.4098 | 0.4529 | 0.9882 | -0.2354 | -0.2802 | 5.949 |
| alibaba2018 | alibaba2018 | clean |  | patchtst | validation_best_capacity_cost | 0.3128 | 0.4098 | 0.4529 | 0.9882 | -0.2354 | -0.2802 | 5.949 |
| alibaba2018 | alibaba2018 | delayed_12 | 0.2 | dlinear | strict_latency_rule | 0.4279 | 0.525 | 1.27 | 0.1422 | 0 | 0 | 0 |
| alibaba2018 | alibaba2018 | delayed_12 | 0.5 | dlinear | validation_best_capacity_cost | 0.4279 | 0.525 | 1.27 | 0.1422 | 0 | 0 | 0 |
| alibaba2018 | alibaba2018 | delayed_12 | 1 | patchtst | validation_best_capacity_cost | 0.4319 | 0.5073 | 0.6864 | 0.9882 | 0.009291 | -0.4596 | 5.949 |
| alibaba2018 | alibaba2018 | delayed_12 |  | patchtst | validation_best_capacity_cost | 0.4319 | 0.5073 | 0.6864 | 0.9882 | 0.009291 | -0.4596 | 5.949 |
| alibaba2018 | alibaba2018 | level_shift | 0.2 | dlinear | strict_latency_rule | 0.7366 | 0.7222 | 0.6242 | 0.1422 | 0 | 0 | 0 |
| alibaba2018 | alibaba2018 | level_shift | 0.5 | race_dlinear | validation_best_capacity_cost | 0.7049 | 0.7036 | 0.5948 | 0.3623 | -0.04304 | -0.04699 | 1.548 |
| alibaba2018 | alibaba2018 | level_shift | 1 | patchtst | validation_best_capacity_cost | 0.3987 | 0.4833 | 0.4632 | 0.9882 | -0.4587 | -0.2579 | 5.949 |
| alibaba2018 | alibaba2018 | level_shift |  | patchtst | validation_best_capacity_cost | 0.3987 | 0.4833 | 0.4632 | 0.9882 | -0.4587 | -0.2579 | 5.949 |
| alibaba2018 | alibaba2018 | missing_30 | 0.2 | dlinear | strict_latency_rule | 0.2862 | 0.4056 | 0.5092 | 0.1422 | 4.561e-04 | -0.004964 | 0 |
| alibaba2018 | alibaba2018 | missing_30 | 0.5 | dlinear | validation_best_capacity_cost | 0.2862 | 0.4056 | 0.5092 | 0.1422 | 4.561e-04 | -0.004964 | 0 |
| alibaba2018 | alibaba2018 | missing_30 | 1 | patchtst | validation_best_capacity_cost | 0.2862 | 0.3911 | 0.4305 | 0.9882 | 6.536e-04 | -0.1588 | 5.949 |
| alibaba2018 | alibaba2018 | missing_30 |  | patchtst | validation_best_capacity_cost | 0.2862 | 0.3911 | 0.4305 | 0.9882 | 6.536e-04 | -0.1588 | 5.949 |
| alibaba2018 | alibaba2018 | missing_variables_30 | 0.2 | dlinear | strict_latency_rule | 0.546 | 0.577 | 1.179 | 0.1422 | -0.001527 | 0.00359 | 0 |
| alibaba2018 | alibaba2018 | missing_variables_30 | 0.5 | dlinear | validation_best_capacity_cost | 0.546 | 0.577 | 1.179 | 0.1422 | -0.001527 | 0.00359 | 0 |
| alibaba2018 | alibaba2018 | missing_variables_30 | 1 | patchtst | metric_outage_rule | 0.521 | 0.5328 | 1.087 | 0.9882 | -0.04726 | -0.07449 | 5.949 |
| alibaba2018 | alibaba2018 | missing_variables_30 |  | patchtst | metric_outage_rule | 0.521 | 0.5328 | 1.087 | 0.9882 | -0.04726 | -0.07449 | 5.949 |
| salesforce_borg | salesforce_borg_256x2048 | burst | 0.2 | dlinear | strict_latency_rule | 0.1225 | 0.2906 | 0.2705 | 0.1502 | 0.002373 | 0.001724 | 0 |
| salesforce_borg | salesforce_borg_256x2048 | burst | 0.5 | race_dlinear | validation_best_capacity_cost | 0.1205 | 0.2894 | 0.2586 | 0.3616 | -0.01447 | -0.04206 | 1.408 |
| salesforce_borg | salesforce_borg_256x2048 | burst | 1 | race_dlinear | validation_best_capacity_cost | 0.1205 | 0.2894 | 0.2586 | 0.3616 | -0.01447 | -0.04206 | 1.408 |
| salesforce_borg | salesforce_borg_256x2048 | burst |  | race_dlinear | validation_best_capacity_cost | 0.1205 | 0.2894 | 0.2586 | 0.3616 | -0.01447 | -0.04206 | 1.408 |
| salesforce_borg | salesforce_borg_256x2048 | clean | 0.2 | dlinear | strict_latency_rule | 0.1177 | 0.2853 | 0.2512 | 0.1502 | 0 | 0 | 0 |
| salesforce_borg | salesforce_borg_256x2048 | clean | 0.5 | race_dlinear | validation_best_capacity_cost | 0.1224 | 0.2887 | 0.2486 | 0.3616 | 0.04028 | -0.01059 | 1.408 |
| salesforce_borg | salesforce_borg_256x2048 | clean | 1 | patchtst | validation_best_capacity_cost | 0.03965 | 0.1298 | 0.1996 | 0.66 | -0.6631 | -0.2056 | 3.395 |
| salesforce_borg | salesforce_borg_256x2048 | clean |  | patchtst | validation_best_capacity_cost | 0.03965 | 0.1298 | 0.1996 | 0.66 | -0.6631 | -0.2056 | 3.395 |
| salesforce_borg | salesforce_borg_256x2048 | delayed_12 | 0.2 | dlinear | strict_latency_rule | 0.1598 | 0.3297 | 0.5353 | 0.1502 | 0 | 0 | 0 |
| salesforce_borg | salesforce_borg_256x2048 | delayed_12 | 0.5 | race_dlinear | validation_best_capacity_cost | 0.1528 | 0.3226 | 0.5201 | 0.3616 | -0.04365 | -0.02839 | 1.408 |
| salesforce_borg | salesforce_borg_256x2048 | delayed_12 | 1 | patchtst | validation_best_capacity_cost | 0.1205 | 0.2684 | 0.4412 | 0.66 | -0.246 | -0.1758 | 3.395 |
| salesforce_borg | salesforce_borg_256x2048 | delayed_12 |  | patchtst | validation_best_capacity_cost | 0.1205 | 0.2684 | 0.4412 | 0.66 | -0.246 | -0.1758 | 3.395 |
| salesforce_borg | salesforce_borg_256x2048 | level_shift | 0.2 | dlinear | strict_latency_rule | 0.2515 | 0.3771 | 0.4182 | 0.1502 | 0 | 0 | 0 |
| salesforce_borg | salesforce_borg_256x2048 | level_shift | 0.5 | race_dlinear | validation_best_capacity_cost | 0.1454 | 0.3148 | 0.3289 | 0.3616 | -0.4217 | -0.2136 | 1.408 |
| salesforce_borg | salesforce_borg_256x2048 | level_shift | 1 | race_dlinear | validation_best_capacity_cost | 0.1454 | 0.3148 | 0.3289 | 0.3616 | -0.4217 | -0.2136 | 1.408 |
| salesforce_borg | salesforce_borg_256x2048 | level_shift |  | race_dlinear | validation_best_capacity_cost | 0.1454 | 0.3148 | 0.3289 | 0.3616 | -0.4217 | -0.2136 | 1.408 |
| salesforce_borg | salesforce_borg_256x2048 | missing_30 | 0.2 | dlinear | strict_latency_rule | 0.03817 | 0.1305 | 0.201 | 0.1502 | -0.003386 | 4.639e-04 | 0 |
| salesforce_borg | salesforce_borg_256x2048 | missing_30 | 0.5 | dlinear | validation_best_capacity_cost | 0.03817 | 0.1305 | 0.201 | 0.1502 | -0.003386 | 4.639e-04 | 0 |
| salesforce_borg | salesforce_borg_256x2048 | missing_30 | 1 | patchtst | validation_best_capacity_cost | 0.03275 | 0.1048 | 0.1708 | 0.66 | -0.145 | -0.1496 | 3.395 |
| salesforce_borg | salesforce_borg_256x2048 | missing_30 |  | patchtst | validation_best_capacity_cost | 0.03275 | 0.1048 | 0.1708 | 0.66 | -0.145 | -0.1496 | 3.395 |
| salesforce_borg | salesforce_borg_256x2048 | missing_variables_30 | 0.2 | dlinear | strict_latency_rule | 0.2531 | 0.3823 | 0.529 | 0.1502 | -0.004476 | 0.00123 | 0 |
| salesforce_borg | salesforce_borg_256x2048 | missing_variables_30 | 0.5 | race_dlinear | validation_best_capacity_cost | 0.2488 | 0.3765 | 0.5214 | 0.3616 | -0.02125 | -0.01326 | 1.408 |
| salesforce_borg | salesforce_borg_256x2048 | missing_variables_30 | 1 | patchtst | metric_outage_rule | 0.197 | 0.2791 | 0.5001 | 0.66 | -0.2249 | -0.05351 | 3.395 |
| salesforce_borg | salesforce_borg_256x2048 | missing_variables_30 |  | patchtst | metric_outage_rule | 0.197 | 0.2791 | 0.5001 | 0.66 | -0.2249 | -0.05351 | 3.395 |
