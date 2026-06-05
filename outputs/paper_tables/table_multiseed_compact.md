| dataset | stress | seeds | best_mse_model | best_capacity_model | lowest_latency_model | mse_vs_capacity_flip | paired_t_p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| alibaba2018 | clean | 5 | race_dlinear | race_dlinear | dlinear | 0 |  |
| alibaba2018 | delayed_12 | 5 | race_dlinear | race_dlinear | dlinear | 0 |  |
| alibaba2018 | level_shift | 5 | race_dlinear | race_dlinear | dlinear | 0 |  |
| alibaba2018 | missing_30 | 5 | race_dlinear | patchtst | dlinear | 1 | 0.001093 |
| alibaba2018 | missing_variables_30 | 5 | patchtst | patchtst | dlinear | 0 |  |
| salesforce_borg | clean | 5 | dlinear | race_dlinear | dlinear | 1 | 7.145e-04 |
| salesforce_borg | delayed_12 | 5 | dlinear | race_dlinear | dlinear | 1 | 0.003728 |
| salesforce_borg | level_shift | 5 | dlinear | race_dlinear | dlinear | 1 | 0.006426 |
| salesforce_borg | missing_30 | 5 | patchtst | patchtst | dlinear | 0 |  |
| salesforce_borg | missing_variables_30 | 5 | patchtst | patchtst | dlinear | 0 |  |
