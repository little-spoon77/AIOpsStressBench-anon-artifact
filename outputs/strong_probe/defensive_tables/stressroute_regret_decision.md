# StressRoute regret probe

This probe reuses existing StressRoute v1/v2 outputs and computes oracle-gap/regret summaries.
It is a defensive analysis only; it does not change the core paper tables.

## Files

- `stressroute_regret_long.csv`
- `stressroute_regret_summary.csv`
- `stressroute_selection_distribution.csv`
- `stressroute_regret_compact.csv`

## Compact view

| dataset                  | objective   |   latency_budget_ms | best_route_policy   |   best_route_capacity_regret | best_fixed_policy   |   best_fixed_capacity_regret |   route_minus_fixed_regret |   oracle_capacity_cost |   route_p95_ms |   route_feasible_rate |
|:-------------------------|:------------|--------------------:|:--------------------|-----------------------------:|:--------------------|-----------------------------:|---------------------------:|-----------------------:|---------------:|----------------------:|
| alibaba2018              | capacity    |                 0.2 | stressroute_v1      |                     0        | fixed_dlinear       |                     0        |                 0          |               0.808667 |       0.141977 |                     1 |
| alibaba2018              | capacity    |                 0.5 | stressroute_v2      |                     0.18826  | fixed_dlinear       |                     0.210721 |                -0.0224608  |               0.681997 |       0.356446 |                     1 |
| alibaba2018              | capacity    |                 1   | stressroute_v1      |                     0.295911 | fixed_patchtst      |                     0.295911 |                 0          |               0.453718 |       0.623876 |                     1 |
| alibaba2018              | capacity    |               nan   | stressroute_v1      |                     0.295911 | fixed_patchtst      |                     0.295911 |                 0          |               0.453718 |       0.623876 |                     1 |
| alibaba2018              | mse         |                 0.2 | stressroute_v1      |                     0        | fixed_dlinear       |                     0        |                 0          |               0.808667 |       0.141977 |                     1 |
| alibaba2018              | mse         |                 0.5 | stressroute_v2      |                     0.191476 | fixed_dlinear       |                     0.194604 |                -0.00312735 |               0.688147 |       0.356446 |                     1 |
| alibaba2018              | mse         |                 1   | stressroute_v2      |                     0.237483 | fixed_patchtst      |                     0.229906 |                 0.00757692 |               0.473756 |       0.623876 |                     1 |
| alibaba2018              | mse         |               nan   | stressroute_v2      |                     0.234225 | fixed_patchtst      |                     0.229906 |                 0.00431891 |               0.473756 |       0.623876 |                     1 |
| salesforce_borg_256x2048 | capacity    |                 0.2 | stressroute_v1      |                     0        | fixed_dlinear       |                     0        |                 0          |               0.368168 |       0.144648 |                     1 |
| salesforce_borg_256x2048 | capacity    |                 0.5 | stressroute_v1      |                     0.292021 | fixed_dlinear       |                     0.393654 |                -0.101633   |               0.261784 |       0.314429 |                     1 |
| salesforce_borg_256x2048 | capacity    |                 1   | stressroute_v1      |                     0.49114  | fixed_patchtst      |                     0.515059 |                -0.0239195  |               0.208948 |       0.517836 |                     1 |
| salesforce_borg_256x2048 | capacity    |               nan   | stressroute_v1      |                     0.49114  | fixed_patchtst      |                     0.515059 |                -0.0239195  |               0.208948 |       0.517836 |                     1 |
| salesforce_borg_256x2048 | mse         |                 0.2 | stressroute_v1      |                     0        | fixed_dlinear       |                     0        |                 0          |               0.368168 |       0.144648 |                     1 |
| salesforce_borg_256x2048 | mse         |                 0.5 | stressroute_v2      |                     0.193309 | fixed_dlinear       |                     0.351103 |                -0.157794   |               0.268205 |       0.39932  |                     1 |
| salesforce_borg_256x2048 | mse         |                 1   | stressroute_v2      |                     0.370836 | fixed_patchtst      |                     0.443475 |                -0.0726384  |               0.21811  |       0.640763 |                     1 |
| salesforce_borg_256x2048 | mse         |               nan   | stressroute_v2      |                     0.371971 | fixed_patchtst      |                     0.443475 |                -0.0715034  |               0.21811  |       0.640763 |                     1 |
