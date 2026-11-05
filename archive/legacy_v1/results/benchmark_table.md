# Official Comparative Benchmark Matrix (ETTh1)

| Model | Parameters | MAE (lower is better) | MSE (lower is better) | Latency (ms/sample) | Bypass / Hit Rate |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **DLinear** | 18,624 | **0.4472** | **0.4497** | 0.031 ms | - |
| **PatchLinear** | 16,992 | **0.4545** | **0.456** | 0.021 ms | - |
| **LSTM** | 95,648 | **0.667** | **0.8285** | 0.147 ms | - |
| **TCN** | 70,304 | **0.7327** | **1.0179** | 0.104 ms | - |
| **EquiTS_NoMem** | 29,087 | **0.5203** | **0.5508** | 0.765 ms | - |
| **EquiTS_RCPQ** | 29,087 | **0.7301** | **1.1539** | 0.33 ms | 99.03 |
