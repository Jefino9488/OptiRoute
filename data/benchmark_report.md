# OptiRoute Benchmark Report

## Model Performance by Capability

### Code

| Model | Accuracy | Avg Latency | Avg Out Tokens | Avg Cost | Routing Rec |
|---|---|---|---|---|---|
| local:qwen-2.5-3b | 0.96 | 9629ms | 139 | $0.000000 | >= 0.91 |
| kimi-k2p7-code | 0.97 | 6433ms | 543 | $0.002841 | >= 0.92 |
| minimax-m3 | 0.97 | 7391ms | 1132 | $0.001399 | >= 0.92 |

### Creative

| Model | Accuracy | Avg Latency | Avg Out Tokens | Avg Cost | Routing Rec |
|---|---|---|---|---|---|
| local:qwen-2.5-3b | 0.64 | 5890ms | 85 | $0.000000 | >= 0.59 |
| kimi-k2p7-code | 0.78 | 7654ms | 616 | $0.003224 | >= 0.73 |
| minimax-m3 | 0.78 | 5701ms | 622 | $0.000788 | >= 0.73 |

### Extraction

| Model | Accuracy | Avg Latency | Avg Out Tokens | Avg Cost | Routing Rec |
|---|---|---|---|---|---|
| local:qwen-2.5-3b | 1.00 | 1833ms | 22 | $0.000000 | >= 0.95 |
| kimi-k2p7-code | 0.84 | 2563ms | 100 | $0.000556 | >= 0.79 |
| minimax-m3 | 0.75 | 3385ms | 182 | $0.000266 | >= 0.70 |

### General_qa

| Model | Accuracy | Avg Latency | Avg Out Tokens | Avg Cost | Routing Rec |
|---|---|---|---|---|---|
| local:qwen-2.5-3b | 0.79 | 3534ms | 50 | $0.000000 | >= 0.74 |
| kimi-k2p7-code | 0.63 | 5023ms | 252 | $0.001324 | >= 0.58 |
| minimax-m3 | 0.73 | 5474ms | 406 | $0.000527 | >= 0.68 |

### Math

| Model | Accuracy | Avg Latency | Avg Out Tokens | Avg Cost | Routing Rec |
|---|---|---|---|---|---|
| local:qwen-2.5-3b | 0.75 | 1755ms | 22 | $0.000000 | >= 0.70 |
| kimi-k2p7-code | 0.85 | 3558ms | 78 | $0.000426 | >= 0.80 |
| minimax-m3 | 0.85 | 3991ms | 208 | $0.000291 | >= 0.80 |

### Reasoning

| Model | Accuracy | Avg Latency | Avg Out Tokens | Avg Cost | Routing Rec |
|---|---|---|---|---|---|
| local:qwen-2.5-3b | 0.69 | 6570ms | 95 | $0.000000 | >= 0.64 |
| kimi-k2p7-code | 0.86 | 8396ms | 710 | $0.003716 | >= 0.81 |
| minimax-m3 | 0.86 | 7625ms | 804 | $0.001007 | >= 0.81 |

### Retrieval

| Model | Accuracy | Avg Latency | Avg Out Tokens | Avg Cost | Routing Rec |
|---|---|---|---|---|---|
| local:qwen-2.5-3b | 0.96 | 822ms | 10 | $0.000000 | >= 0.91 |
| kimi-k2p7-code | 0.48 | 1153ms | 54 | $0.000297 | DO NOT ROUTE |
| minimax-m3 | 0.64 | 2397ms | 70 | $0.000125 | >= 0.59 |

### Translation

| Model | Accuracy | Avg Latency | Avg Out Tokens | Avg Cost | Routing Rec |
|---|---|---|---|---|---|
| local:qwen-2.5-3b | 0.46 | 893ms | 10 | $0.000000 | DO NOT ROUTE |
| kimi-k2p7-code | 0.30 | 5441ms | 185 | $0.000983 | DO NOT ROUTE |
| minimax-m3 | 0.16 | 3604ms | 255 | $0.000347 | DO NOT ROUTE |

## Failure Patterns

- **kimi-k2p7-code**: fails on retrieval, translation
- **minimax-m3**: fails on translation
- **local:qwen-2.5-3b**: fails on translation
