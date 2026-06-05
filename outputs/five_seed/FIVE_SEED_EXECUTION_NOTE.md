# Five-seed stability execution note

## Scope

This run is an artifact-level defense for seed stability. It does not overwrite the current paper PDF or paper tables.

- Datasets: Alibaba 2018 and Salesforce/Borg 256x2048.
- Scenarios: clean, missing_30, missing_variables_30, delayed_12, level_shift.
- Seeds: 42, 2024, 2025, 2026, 2027.
- Core lightweight pool: dlinear, race_dlinear, patchtst.
- Official pool: existing single-seed probe only; not treated as a five-seed claim.

## Completion

- Core rows with complete 5 seeds: 30 model-dataset-stress rows.
- MSE-vs-capacity disagreements: 4 / 10 core dataset-stress settings.
- MSE-vs-capacity disagreements whose capacity gap is outside one standard-deviation bands: 4 / 10 core dataset-stress settings.
- Accuracy-vs-latency disagreement is reported separately because it is often driven by DLinear's structurally low latency.
- Key columns were checked locally: dataset, stress, model, seed, mse, mae, capacity_cost, latency_p95_ms contain no NaN in the merged long table.

## Main interpretation boundary

Use MSE-vs-capacity disagreement as the non-trivial ranking-reversal evidence. Do not use the combined objective_disagreement=1.0 metric as the flagship claim because it folds in latency differences that are largely structural.

## Core winners

| dataset         | stress               | best_mse_model   | best_capacity_model   | lowest_latency_model   | mse_vs_capacity_disagreement   | accuracy_vs_latency_disagreement   |
|:----------------|:---------------------|:-----------------|:----------------------|:-----------------------|:-------------------------------|:-----------------------------------|
| alibaba2018     | clean                | race_dlinear     | race_dlinear          | dlinear                | False                          | True                               |
| alibaba2018     | delayed_12           | race_dlinear     | race_dlinear          | dlinear                | False                          | True                               |
| alibaba2018     | level_shift          | race_dlinear     | race_dlinear          | dlinear                | False                          | True                               |
| alibaba2018     | missing_30           | race_dlinear     | patchtst              | dlinear                | True                           | True                               |
| alibaba2018     | missing_variables_30 | patchtst         | patchtst              | dlinear                | False                          | True                               |
| salesforce_borg | clean                | dlinear          | race_dlinear          | dlinear                | True                           | False                              |
| salesforce_borg | delayed_12           | dlinear          | race_dlinear          | dlinear                | True                           | False                              |
| salesforce_borg | level_shift          | dlinear          | race_dlinear          | dlinear                | True                           | False                              |
| salesforce_borg | missing_30           | patchtst         | patchtst              | dlinear                | False                          | True                               |
| salesforce_borg | missing_variables_30 | patchtst         | patchtst              | dlinear                | False                          | True                               |

## Gap outside one-standard-deviation bands

| dataset         | stress               | best_mse_model   | best_capacity_model   | mse_vs_capacity_disagreement   |   capacity_gap | capacity_gap_outside_1std   |     mse_gap | mse_gap_outside_1std   |
|:----------------|:---------------------|:-----------------|:----------------------|:-------------------------------|---------------:|:----------------------------|------------:|:-----------------------|
| alibaba2018     | clean                | race_dlinear     | race_dlinear          | False                          |     0.00596321 | False                       | 0.00124476  | True                   |
| alibaba2018     | delayed_12           | race_dlinear     | race_dlinear          | False                          |     0.0111434  | False                       | 0.00212842  | False                  |
| alibaba2018     | level_shift          | race_dlinear     | race_dlinear          | False                          |     0.00680937 | False                       | 0.00138717  | False                  |
| alibaba2018     | missing_30           | race_dlinear     | patchtst              | True                           |     0.0792786  | True                        | 0.00148211  | True                   |
| alibaba2018     | missing_variables_30 | patchtst         | patchtst              | False                          |     0.344719   | True                        | 0.0995627   | True                   |
| salesforce_borg | clean                | dlinear          | race_dlinear          | True                           |     0.0101291  | True                        | 0.000151065 | False                  |
| salesforce_borg | delayed_12           | dlinear          | race_dlinear          | True                           |     0.0184904  | True                        | 0.000631531 | False                  |
| salesforce_borg | level_shift          | dlinear          | race_dlinear          | True                           |     0.00803224 | True                        | 0.000244695 | False                  |
| salesforce_borg | missing_30           | patchtst         | patchtst              | False                          |     0.00214114 | False                       | 0.00320412  | True                   |
| salesforce_borg | missing_variables_30 | patchtst         | patchtst              | False                          |     0.131133   | True                        | 0.0684617   | True                   |

## Paper-use recommendation

If this result is moved into the paper, write: in the core lightweight five-seed artifact, MSE and capacity winners differ in 4/10 dataset-stress settings; all four capacity gaps are outside one standard-deviation bands. Separately, accuracy-vs-latency disagreement is a deployment-cost observation, not the main novelty claim.