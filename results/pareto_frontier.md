# EquiTS: Empirical Accuracy vs. Latency Pareto Frontier (ETTh1)

Evaluation of **EquiTS** with Pre-populated Prototype Memory ($K=512$) on the held-out ETTh1 test split ($L=96, H=96$).
Demonstrates the monotonic trade-off governed by the quotienting threshold $\tau_{\mathrm{key}}$:

| Threshold ($\tau_{\mathrm{key}}$) | MAE (lower is better) | MSE (lower is better) | Latency (ms/sample) | Speedup Gain | Bypass Hit Rate |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **$0.0$ (No Memory)** | **0.5201** | **0.5424** | 0.689 ms | **1.00x** | 0.0% |
| **$1.0 \times 10^{-7}$** | 0.5630 | 0.6677 | 0.635 ms | **1.09x** | 18.3% |
| **$3.0 \times 10^{-7}$** | 0.6051 | 0.7877 | 0.544 ms | **1.27x** | 38.7% |
| **$7.0 \times 10^{-7}$** | 0.6382 | 0.8809 | 0.491 ms | **1.40x** | 54.0% |
| **$1.5 \times 10^{-6}$** | 0.6842 | 1.0136 | 0.392 ms | **1.76x** | 74.2% |
| **$3.0 \times 10^{-6}$** | 0.7197 | 1.1119 | 0.309 ms | **2.23x** | 91.4% |
| **$7.0 \times 10^{-6}$** | 0.7347 | 1.1534 | 0.275 ms | **2.50x** | 97.9% |
| **$1.5 \times 10^{-5}$** | 0.7362 | 1.1571 | **0.257 ms** | **2.68x** | **98.8%** |

### Benchmark Context
* **LSTM (Seq2Seq):** MAE: $0.6670$, Latency: $0.147$ ms
* **TCN (Dilated Causal CNN):** MAE: $0.7327$, Latency: $0.104$ ms
* At $\tau = 3.0 \times 10^{-7}$, EquiTS achieves **$0.6051$ MAE** (substantially outperforming both LSTM and TCN) with a **$1.27\times$ speedup** and **$38.7\%$ bypass**.
* At $\tau = 1.5 \times 10^{-5}$, EquiTS achieves maximum acceleration (**$2.68\times$ speedup**, **$0.257$ ms/sample**) while bypassing **$98.8\%$** of tokens.
