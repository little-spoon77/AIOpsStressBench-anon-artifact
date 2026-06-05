# Expanded official-model multi-seed probe

This table combines existing seed-42 official runs with any new seed-2025/2026 CSVs under `outputs/strong_probe/expanded_multiseed`.
Incomplete rows are kept in the CSV but excluded from the winner summary.

## Complete winner rows

| dataset                  | stress               |   models_with_3_seeds | best_mse_model               | best_capacity_model          | lowest_latency_model         | objective_disagreement   |
|:-------------------------|:---------------------|----------------------:|:-----------------------------|:-----------------------------|:-----------------------------|:-------------------------|
| alibaba2018              | clean                |                     3 | official_timemixer           | official_timemixer           | official_patchtst            | True                     |
| alibaba2018              | delayed_12           |                     3 | official_itransformer_native | official_patchtst            | official_patchtst            | True                     |
| alibaba2018              | missing_variables_30 |                     3 | official_timemixer           | official_timemixer           | official_patchtst            | True                     |
| salesforce_borg_256x2048 | clean                |                     3 | official_timemixer           | official_timemixer           | official_patchtst            | True                     |
| salesforce_borg_256x2048 | delayed_12           |                     3 | official_timemixer           | official_timemixer           | official_itransformer_native | True                     |
| salesforce_borg_256x2048 | missing_variables_30 |                     3 | official_itransformer_native | official_itransformer_native | official_itransformer_native | False                    |
