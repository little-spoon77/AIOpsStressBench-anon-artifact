| source | latency_budget_ms | fixed_patchtst_feasible_rate | fixed_patchtst_mean_cost | v1_selected_models | v1_mean_cost | v1_cost_vs_dlinear | v2_selected_models | v2_mean_cost | v2_cost_vs_dlinear | v2_mean_p95_ms | v2_regret | v2_oracle_gap | oracle_mean_cost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| alibaba2018 | 0.2 | 0 | 0.5953 | dlinear:6 | 0.8087 | 0 | dlinear:6 | 0.8087 | 0 | 0.142 | 0 | 0 | 0.8087 |
| alibaba2018 | 0.5 | 0 | 0.5953 | race_dlinear:4, dlinear:2 | 0.81 | 4.358e-04 | dlinear:3, race_dlinear:3 | 0.7982 | -0.01665 | 0.3564 | 0.1162 | 0.1883 | 0.682 |
| alibaba2018 | 1 | 1 | 0.5953 | patchtst:6 | 0.5953 | -0.2545 | dlinear:3, patchtst:2, race_dlinear:1 | 0.6474 | -0.1758 | 0.6239 | 0.1937 | 0.4244 | 0.4537 |
| salesforce_borg | 0.2 | 0 | 0.3164 | dlinear:6 | 0.3682 | 0 | dlinear:6 | 0.3682 | 0 | 0.1446 | 0 | 0 | 0.3682 |
| salesforce_borg | 0.5 | 0 | 0.3164 | race_dlinear:4, dlinear:2 | 0.347 | -0.05952 | race_dlinear:4, dlinear:2 | 0.3464 | -0.05945 | 0.3569 | 0.08466 | 0.2933 | 0.2618 |
| salesforce_borg | 1 | 1 | 0.3164 | patchtst:4, dlinear:1, race_dlinear:1 | 0.3108 | -0.1529 | dlinear:2, patchtst:2, race_dlinear:2 | 0.3247 | -0.1181 | 0.6408 | 0.1157 | 0.5468 | 0.2089 |
