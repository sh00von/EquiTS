# Official Comparative Benchmark Matrix (ETTm1)

Evaluated on ETTm1 ($L=96, H=96$) under standardized chronological splits:

| Model | Parameters | MAE (lower is better) | MSE (lower is better) | Latency (ms/sample) | Bypass / Hit Rate |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **DLinear** | 18,624 | **0.3939** | **0.3717** | 0.027 ms | - |
| **PatchLinear** | 16,992 | **0.3886** | **0.3659** | 0.023 ms | - |
| **LSTM** | 95,648 | **0.5638** | **0.6242** | 0.197 ms | - |
| **TCN** | 70,304 | **0.5696** | **0.6409** | 0.097 ms | - |
| **EquiTS_NoMem** | 29,087 | **0.4593** | **0.4717** | 0.619 ms | - |
| **EquiTS_RCPQ** | 29,087 | **0.6715** | **1.0163** | 0.248 ms | 99.77 |
