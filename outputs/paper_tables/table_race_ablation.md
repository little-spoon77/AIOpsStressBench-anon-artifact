| run | model | mse | mae | latency_p95_ms | capacity_cost | params | max_memory_mb |
| --- | --- | --- | --- | --- | --- | --- | --- |
| base | dlinear | 0.005283 | 0.01077 | 0.2008 | 3.032e+10 | 2330 | 16 |
| base | race_dlinear_nomask | 0.005592 | 0.01093 | 0.1977 | 2.678e+10 | 4660 | 16 |
| base | race_dlinear | 0.00561 | 0.01009 | 0.2896 | 2.899e+10 | 4660 | 16 |
| no_calibration | race_dlinear | 0.005314 | 0.01106 | 0.3162 | 3.207e+10 | 4660 | 16 |
| no_train_stress | race_dlinear | 0.1156 | 0.05982 | 0.3132 | 2.957e+11 | 4660 | 16 |
