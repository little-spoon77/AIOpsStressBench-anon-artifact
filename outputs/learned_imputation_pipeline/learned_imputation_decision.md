# Learned Imputation Pipeline Decision

- Scope: bounded learned mask-aware imputation baseline, not SAITS/BRITS/PyPOTS.
- Data: Alibaba 2018 and Salesforce/Borg.
- Stress: missing_30 and missing_variables_30.
- Forecasters: DLinear and PatchTST-style.

## Summary

| dataset         | stress               | forecaster   | best_imputation_by_capacity   |   learned_vs_none_capacity_pct |   learned_vs_none_mse_pct |   learned_capacity_cost |   none_capacity_cost |   learned_mse |   none_mse |
|:----------------|:---------------------|:-------------|:------------------------------|-------------------------------:|--------------------------:|------------------------:|---------------------:|--------------:|-----------:|
| alibaba2018     | missing_30           | dlinear      | mean                          |                       -57.2242 |                  -3.18664 |                 6.8914  |             16.1105  |     0.984916  |  1.01733   |
| alibaba2018     | missing_30           | patchtst     | mean                          |                       -79.3175 |                 -13.7629  |                 2.66824 |             12.9009  |     0.336212  |  0.389869  |
| alibaba2018     | missing_variables_30 | dlinear      | learned_mask_imputer          |                       -57.1414 |                  -4.77745 |                 6.58151 |             15.3563  |     0.997223  |  1.04726   |
| alibaba2018     | missing_variables_30 | patchtst     | learned_mask_imputer          |                       -81.7023 |                 -26.0887  |                 3.2026  |             17.5028  |     0.408538  |  0.552741  |
| salesforce_borg | missing_30           | dlinear      | mean                          |                       -56.3428 |                 -20.2552  |                 2.49649 |              5.71838 |     0.171589  |  0.215173  |
| salesforce_borg | missing_30           | patchtst     | mean                          |                       -50.7768 |                 -20.0808  |                 1.00073 |              2.03305 |     0.0706131 |  0.0883557 |
| salesforce_borg | missing_variables_30 | dlinear      | learned_mask_imputer          |                       -57.1256 |                 -34.7058  |                 2.66326 |              6.21178 |     0.201971  |  0.309324  |
| salesforce_borg | missing_variables_30 | patchtst     | learned_mask_imputer          |                       -73.6427 |                 -50.4266  |                 1.313   |              4.98153 |     0.137895  |  0.278162  |

## Decision

- Learned imputation improves at least one pipeline by capacity: yes.
- Channel-outage rows prefer learned imputation by capacity winner: yes.
- Recommended paper use: artifact-first; add a short imputation-scope sentence only if the result is stable and space allows.

## Interpretation

- This is a bounded learned imputation baseline, not a complete missing-data model benchmark.
- This result should not be described as solving telemetry outage; it indicates that preprocessing can change the deployment pipeline choice.
- If learned imputation is the best pipeline in several rows, the benchmark can report preprocessing as a deployment choice.