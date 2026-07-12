# OptiRoute Benchmark Report

## Model Performance by Capability

### Code

| Model | Accuracy | Avg Latency | Avg Out Tokens | Avg Cost | Routing Rec |
|---|---|---|---|---|---|
| kimi-k2p7-code | 0.97 | 4171ms | 603 | $0.004094 | >= 0.92 |
| local:qwen-2.5-3b | 0.93 | 1559ms | 159 | $0.000000 | >= 0.88 |
| minimax-m3 | 0.97 | 6920ms | 1217 | $0.002670 | >= 0.92 |

### Creative

| Model | Accuracy | Avg Latency | Avg Out Tokens | Avg Cost | Routing Rec |
|---|---|---|---|---|---|
| kimi-k2p7-code | 0.72 | 3287ms | 286 | $0.001508 | >= 0.67 |
| local:qwen-2.5-3b | 0.60 | 1313ms | 136 | $0.000000 | >= 0.55 |
| minimax-m3 | 0.74 | 3700ms | 266 | $0.000357 | >= 0.69 |

### Extraction

| Model | Accuracy | Avg Latency | Avg Out Tokens | Avg Cost | Routing Rec |
|---|---|---|---|---|---|
| kimi-k2p7-code | 0.70 | 1467ms | 77 | $0.000440 | >= 0.65 |
| local:qwen-2.5-3b | 0.96 | 172ms | 15 | $0.000000 | >= 0.91 |
| minimax-m3 | 0.50 | 1855ms | 79 | $0.000138 | >= 0.45 |

### General_qa

| Model | Accuracy | Avg Latency | Avg Out Tokens | Avg Cost | Routing Rec |
|---|---|---|---|---|---|
| kimi-k2p7-code | 0.65 | 2959ms | 147 | $0.000783 | >= 0.60 |
| local:qwen-2.5-3b | 0.80 | 528ms | 53 | $0.000000 | >= 0.75 |
| minimax-m3 | 0.74 | 2372ms | 284 | $0.000377 | >= 0.69 |

### Math

| Model | Accuracy | Avg Latency | Avg Out Tokens | Avg Cost | Routing Rec |
|---|---|---|---|---|---|
| kimi-k2p7-code | 0.90 | 1190ms | 75 | $0.000526 | >= 0.85 |
| local:qwen-2.5-3b | 0.75 | 309ms | 30 | $0.000000 | >= 0.70 |
| minimax-m3 | 0.90 | 1885ms | 213 | $0.000502 | >= 0.85 |

### Reasoning

| Model | Accuracy | Avg Latency | Avg Out Tokens | Avg Cost | Routing Rec |
|---|---|---|---|---|---|
| kimi-k2p7-code | 0.83 | 5256ms | 520 | $0.003537 | >= 0.78 |
| local:qwen-2.5-3b | 0.70 | 1043ms | 108 | $0.000000 | >= 0.65 |
| minimax-m3 | 0.85 | 5868ms | 715 | $0.001587 | >= 0.80 |

### Retrieval

| Model | Accuracy | Avg Latency | Avg Out Tokens | Avg Cost | Routing Rec |
|---|---|---|---|---|---|
| kimi-k2p7-code | 0.48 | 1366ms | 24 | $0.000140 | DO NOT ROUTE |
| local:qwen-2.5-3b | 0.96 | 111ms | 9 | $0.000000 | >= 0.91 |
| minimax-m3 | 0.64 | 1048ms | 61 | $0.000110 | >= 0.59 |

### Translation

| Model | Accuracy | Avg Latency | Avg Out Tokens | Avg Cost | Routing Rec |
|---|---|---|---|---|---|
| kimi-k2p7-code | 0.30 | 1856ms | 66 | $0.000365 | DO NOT ROUTE |
| local:qwen-2.5-3b | 0.35 | 235ms | 20 | $0.000000 | DO NOT ROUTE |
| minimax-m3 | 0.30 | 1809ms | 73 | $0.000126 | DO NOT ROUTE |

