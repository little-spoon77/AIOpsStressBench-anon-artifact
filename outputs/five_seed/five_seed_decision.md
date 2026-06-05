# Five-seed decision report

Expected complete core rows: 30; observed complete core rows: 30.
Official probe rows present: 28.
Core MSE-vs-capacity disagreements: 4/10.
Core MSE-vs-capacity disagreements with capacity winner gap outside one standard deviation: 4/10.

## Objective disagreement stability

| pool | dataset | stress | seeds | mse_vs_capacity_disagreement_rate | accuracy_vs_latency_disagreement_rate | disagreement_rate |
| --- | --- | --- | --- | --- | --- | --- |
| core_lightweight | alibaba2018 | clean | 5 | 0.4 | 1 | 1 |
| core_lightweight | alibaba2018 | delayed_12 | 5 | 0.6 | 1 | 1 |
| core_lightweight | alibaba2018 | level_shift | 5 | 0.4 | 0.8 | 1 |
| core_lightweight | alibaba2018 | missing_30 | 5 | 1 | 1 | 1 |
| core_lightweight | alibaba2018 | missing_variables_30 | 5 | 0 | 1 | 1 |
| core_lightweight | salesforce_borg | clean | 5 | 0.4 | 0.6 | 1 |
| core_lightweight | salesforce_borg | delayed_12 | 5 | 0.8 | 0.2 | 1 |
| core_lightweight | salesforce_borg | level_shift | 5 | 1 | 0 | 1 |
| core_lightweight | salesforce_borg | missing_30 | 5 | 0.4 | 1 | 1 |
| core_lightweight | salesforce_borg | missing_variables_30 | 5 | 0 | 1 | 1 |
| official_probe | alibaba2018 | clean | 1 | 0 | 0 | 0 |
| official_probe | alibaba2018 | delayed_12 | 1 | 1 | 1 | 1 |
| official_probe | alibaba2018 | level_shift | 1 | 1 | 1 | 1 |
| official_probe | alibaba2018 | missing_30 | 1 | 1 | 1 | 1 |
| official_probe | alibaba2018 | missing_variables_30 | 1 | 1 | 1 | 1 |
| official_probe | salesforce_borg | clean | 1 | 0 | 1 | 1 |
| official_probe | salesforce_borg | delayed_12 | 1 | 1 | 1 | 1 |
| official_probe | salesforce_borg | missing_30 | 1 | 1 | 1 | 1 |
| official_probe | salesforce_borg | missing_variables_30 | 1 | 0 | 0 | 0 |

Note: `objective_disagreement` combines MSE-vs-capacity disagreement and accuracy-vs-latency disagreement. The latter is often driven by the structurally low latency of DLinear and should not be used as the main ranking-reversal claim.

## Winner summary

| pool | dataset | stress | models | min_seeds | complete_5_seed_models | best_mse_model | best_capacity_model | lowest_latency_model | mse_vs_capacity_disagreement | accuracy_vs_latency_disagreement | objective_disagreement |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| core_lightweight | alibaba2018 | clean | 3 | 5 | 3 | race_dlinear | race_dlinear | dlinear | False | True | True |
| core_lightweight | alibaba2018 | delayed_12 | 3 | 5 | 3 | race_dlinear | race_dlinear | dlinear | False | True | True |
| core_lightweight | alibaba2018 | level_shift | 3 | 5 | 3 | race_dlinear | race_dlinear | dlinear | False | True | True |
| core_lightweight | alibaba2018 | missing_30 | 3 | 5 | 3 | race_dlinear | patchtst | dlinear | True | True | True |
| core_lightweight | alibaba2018 | missing_variables_30 | 3 | 5 | 3 | patchtst | patchtst | dlinear | False | True | True |
| core_lightweight | salesforce_borg | clean | 3 | 5 | 3 | dlinear | race_dlinear | dlinear | True | False | True |
| core_lightweight | salesforce_borg | delayed_12 | 3 | 5 | 3 | dlinear | race_dlinear | dlinear | True | False | True |
| core_lightweight | salesforce_borg | level_shift | 3 | 5 | 3 | dlinear | race_dlinear | dlinear | True | False | True |
| core_lightweight | salesforce_borg | missing_30 | 3 | 5 | 3 | patchtst | patchtst | dlinear | False | True | True |
| core_lightweight | salesforce_borg | missing_variables_30 | 3 | 5 | 3 | patchtst | patchtst | dlinear | False | True | True |
| official_probe | alibaba2018 | clean | 3 | 1 | 0 | official_itransformer_native | official_itransformer_native | official_itransformer_native | False | False | False |
| official_probe | alibaba2018 | delayed_12 | 3 | 1 | 0 | official_timemixer | official_patchtst | official_patchtst | True | True | True |
| official_probe | alibaba2018 | level_shift | 3 | 1 | 0 | official_timemixer | official_itransformer_native | official_itransformer_native | True | True | True |
| official_probe | alibaba2018 | missing_30 | 3 | 1 | 0 | official_timemixer | official_patchtst | official_patchtst | True | True | True |
| official_probe | alibaba2018 | missing_variables_30 | 3 | 1 | 0 | official_timemixer | official_patchtst | official_itransformer_native | True | True | True |
| official_probe | salesforce_borg | clean | 3 | 1 | 0 | official_timemixer | official_timemixer | official_patchtst | False | True | True |
| official_probe | salesforce_borg | delayed_12 | 3 | 1 | 0 | official_timemixer | official_patchtst | official_itransformer_native | True | True | True |
| official_probe | salesforce_borg | level_shift | 1 | 1 | 0 | official_timemixer | official_timemixer | official_timemixer | False | False | False |
| official_probe | salesforce_borg | missing_30 | 3 | 1 | 0 | official_timemixer | official_patchtst | official_itransformer_native | True | True | True |
| official_probe | salesforce_borg | missing_variables_30 | 3 | 1 | 0 | official_itransformer_native | official_itransformer_native | official_itransformer_native | False | False | False |

## Gap outside one-standard-deviation bands

| pool | dataset | stress | best_mse_model | second_mse_model | mse_gap | mse_gap_outside_1std | best_capacity_model | second_capacity_model | capacity_gap | capacity_gap_outside_1std | mse_vs_capacity_disagreement |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| core_lightweight | alibaba2018 | clean | race_dlinear | dlinear | 0.001245 | True | race_dlinear | dlinear | 0.005963 | False | False |
| core_lightweight | alibaba2018 | delayed_12 | race_dlinear | dlinear | 0.002128 | False | race_dlinear | dlinear | 0.01114 | False | False |
| core_lightweight | alibaba2018 | level_shift | race_dlinear | dlinear | 0.001387 | False | race_dlinear | patchtst | 0.006809 | False | False |
| core_lightweight | alibaba2018 | missing_30 | race_dlinear | dlinear | 0.001482 | True | patchtst | race_dlinear | 0.07928 | True | True |
| core_lightweight | alibaba2018 | missing_variables_30 | patchtst | race_dlinear | 0.09956 | True | patchtst | race_dlinear | 0.3447 | True | False |
| core_lightweight | salesforce_borg | clean | dlinear | race_dlinear | 0.0001511 | False | race_dlinear | dlinear | 0.01013 | True | True |
| core_lightweight | salesforce_borg | delayed_12 | dlinear | race_dlinear | 0.0006315 | False | race_dlinear | dlinear | 0.01849 | True | True |
| core_lightweight | salesforce_borg | level_shift | dlinear | race_dlinear | 0.0002447 | False | race_dlinear | dlinear | 0.008032 | True | True |
| core_lightweight | salesforce_borg | missing_30 | patchtst | dlinear | 0.003204 | True | patchtst | race_dlinear | 0.002141 | False | False |
| core_lightweight | salesforce_borg | missing_variables_30 | patchtst | race_dlinear | 0.06846 | True | patchtst | race_dlinear | 0.1311 | True | False |
| official_probe | alibaba2018 | clean | official_itransformer_native | official_patchtst | 0.0003654 | True | official_itransformer_native | official_patchtst | 4.706e-05 | True | False |
| official_probe | alibaba2018 | delayed_12 | official_timemixer | official_itransformer_native | 0.003691 | True | official_patchtst | official_itransformer_native | 0.01271 | True | True |
| official_probe | alibaba2018 | level_shift | official_timemixer | official_itransformer_native | 0.004899 | True | official_itransformer_native | official_timemixer | 0.00243 | True | True |
| official_probe | alibaba2018 | missing_30 | official_timemixer | official_patchtst | 0.001038 | True | official_patchtst | official_itransformer_native | 0.01478 | True | True |
| official_probe | alibaba2018 | missing_variables_30 | official_timemixer | official_patchtst | 0.007548 | True | official_patchtst | official_timemixer | 0.004996 | True | True |
| official_probe | salesforce_borg | clean | official_timemixer | official_patchtst | 0.0005738 | True | official_timemixer | official_itransformer_native | 0.003003 | True | False |
| official_probe | salesforce_borg | delayed_12 | official_timemixer | official_patchtst | 0.001187 | True | official_patchtst | official_timemixer | 0.005635 | True | True |
| official_probe | salesforce_borg | missing_30 | official_timemixer | official_patchtst | 0.002414 | True | official_patchtst | official_timemixer | 0.00489 | True | True |
| official_probe | salesforce_borg | missing_variables_30 | official_itransformer_native | official_timemixer | 0.01794 | True | official_itransformer_native | official_timemixer | 0.009697 | True | False |

## Decision

Core lightweight 5-seed evidence is complete. Use MSE-vs-capacity disagreement as the primary non-trivial seed-stability defense, and keep accuracy-vs-latency disagreement as a deployment-cost observation rather than the main novelty claim.