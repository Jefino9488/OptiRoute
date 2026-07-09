# OptiRoute Benchmark Report

## Model Performance by Capability

### Code

| Model | Accuracy | Avg Latency | Avg Out Tokens | Avg Cost | Routing Rec |
|---|---|---|---|---|---|
| local:qwen-2.5-3b | 0.96 | 9629ms | 139 | $0.000000 | >= 0.91 |

### Creative

| Model | Accuracy | Avg Latency | Avg Out Tokens | Avg Cost | Routing Rec |
|---|---|---|---|---|---|
| local:qwen-2.5-3b | 0.64 | 5890ms | 85 | $0.000000 | >= 0.59 |

### Extraction

| Model | Accuracy | Avg Latency | Avg Out Tokens | Avg Cost | Routing Rec |
|---|---|---|---|---|---|
| local:qwen-2.5-3b | 1.00 | 1833ms | 22 | $0.000000 | >= 0.95 |

### General_qa

| Model | Accuracy | Avg Latency | Avg Out Tokens | Avg Cost | Routing Rec |
|---|---|---|---|---|---|
| local:qwen-2.5-3b | 0.79 | 3534ms | 50 | $0.000000 | >= 0.74 |

### Math

| Model | Accuracy | Avg Latency | Avg Out Tokens | Avg Cost | Routing Rec |
|---|---|---|---|---|---|
| local:qwen-2.5-3b | 0.75 | 1755ms | 22 | $0.000000 | >= 0.70 |

### Reasoning

| Model | Accuracy | Avg Latency | Avg Out Tokens | Avg Cost | Routing Rec |
|---|---|---|---|---|---|
| local:qwen-2.5-3b | 0.69 | 6570ms | 95 | $0.000000 | >= 0.64 |

### Retrieval

| Model | Accuracy | Avg Latency | Avg Out Tokens | Avg Cost | Routing Rec |
|---|---|---|---|---|---|
| local:qwen-2.5-3b | 0.96 | 822ms | 10 | $0.000000 | >= 0.91 |

### Translation

| Model | Accuracy | Avg Latency | Avg Out Tokens | Avg Cost | Routing Rec |
|---|---|---|---|---|---|
| local:qwen-2.5-3b | 0.46 | 893ms | 10 | $0.000000 | DO NOT ROUTE |

## Failure Patterns

- **local:qwen-2.5-3b**: fails on translation
