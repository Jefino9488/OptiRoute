# OptiRoute Benchmark Report

## Model Performance by Capability

### Code

| Model | Accuracy | Avg Latency | Avg Out Tokens | Avg Cost | Routing Rec |
|---|---|---|---|---|---|
| kimi-k2p7-code | 0.97 | 7119ms | 570 | $0.000000 | >= 0.92 |
| local:qwen-2.5-3b | 0.96 | 9593ms | 139 | $0.000000 | >= 0.91 |
| minimax-m3 | 0.91 | 6459ms | 979 | $0.000000 | >= 0.86 |

### Creative

| Model | Accuracy | Avg Latency | Avg Out Tokens | Avg Cost | Routing Rec |
|---|---|---|---|---|---|
| kimi-k2p7-code | 0.72 | 6667ms | 510 | $0.000000 | >= 0.67 |
| local:qwen-2.5-3b | 0.60 | 7694ms | 109 | $0.000000 | >= 0.55 |
| minimax-m3 | 0.62 | 6168ms | 595 | $0.000000 | >= 0.57 |

### Extraction

| Model | Accuracy | Avg Latency | Avg Out Tokens | Avg Cost | Routing Rec |
|---|---|---|---|---|---|
| kimi-k2p7-code | 0.46 | 2489ms | 112 | $0.000000 | DO NOT ROUTE |
| local:qwen-2.5-3b | 0.96 | 1352ms | 15 | $0.000000 | >= 0.91 |
| minimax-m3 | 0.64 | 2061ms | 204 | $0.000000 | >= 0.59 |

### General_qa

| Model | Accuracy | Avg Latency | Avg Out Tokens | Avg Cost | Routing Rec |
|---|---|---|---|---|---|
| kimi-k2p7-code | 0.71 | 4313ms | 255 | $0.000000 | >= 0.66 |
| local:qwen-2.5-3b | 0.85 | 3342ms | 47 | $0.000000 | >= 0.80 |
| minimax-m3 | 0.69 | 4590ms | 378 | $0.000000 | >= 0.64 |

### Math

| Model | Accuracy | Avg Latency | Avg Out Tokens | Avg Cost | Routing Rec |
|---|---|---|---|---|---|
| kimi-k2p7-code | 0.85 | 2059ms | 75 | $0.000000 | >= 0.80 |
| local:qwen-2.5-3b | 0.75 | 1753ms | 22 | $0.000000 | >= 0.70 |
| minimax-m3 | 0.85 | 2377ms | 206 | $0.000000 | >= 0.80 |

### Reasoning

| Model | Accuracy | Avg Latency | Avg Out Tokens | Avg Cost | Routing Rec |
|---|---|---|---|---|---|
| kimi-k2p7-code | 0.83 | 7871ms | 586 | $0.000000 | >= 0.78 |
| local:qwen-2.5-3b | 0.70 | 6790ms | 96 | $0.000000 | >= 0.65 |
| minimax-m3 | 0.87 | 6589ms | 807 | $0.000000 | >= 0.82 |

### Retrieval

| Model | Accuracy | Avg Latency | Avg Out Tokens | Avg Cost | Routing Rec |
|---|---|---|---|---|---|
| kimi-k2p7-code | 0.64 | 1429ms | 60 | $0.000000 | >= 0.59 |
| local:qwen-2.5-3b | 0.96 | 925ms | 10 | $0.000000 | >= 0.91 |
| minimax-m3 | 0.32 | 2165ms | 87 | $0.000000 | DO NOT ROUTE |

### Translation

| Model | Accuracy | Avg Latency | Avg Out Tokens | Avg Cost | Routing Rec |
|---|---|---|---|---|---|
| kimi-k2p7-code | 0.29 | 4025ms | 276 | $0.000000 | DO NOT ROUTE |
| local:qwen-2.5-3b | 0.46 | 1109ms | 12 | $0.000000 | DO NOT ROUTE |
| minimax-m3 | 0.17 | 2612ms | 282 | $0.000000 | DO NOT ROUTE |

## Failure Patterns

- **kimi-k2p7-code**: fails on extraction, translation
- **local:qwen-2.5-3b**: fails on translation
- **minimax-m3**: fails on retrieval, translation
