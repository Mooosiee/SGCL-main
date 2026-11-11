# Benchmark Comparison Summary

## Dataset: amazon-beauty

| Model | Seeds | Recall@20 | NDCG@20 | Recall@50 | NDCG@50 | Hit@20 | MRR@20 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| SGCL | 1 | 0.1431 | 0.0689 | 0.2121 | 0.0831 | 0.1531 | 0.0503 |
| FS-SGCL | 1 | 0.1447 | 0.0711 | 0.2150 | 0.0855 | 0.1549 | 0.0526 |

### Efficiency — amazon-beauty

| Model | Device | Params | Best Epoch | Wall Time (s) |
| :--- | :--- | :--- | :--- | :--- |
| SGCL | cpu | 2,205,824 | 56.0 | 3171.8 |
| FS-SGCL | cpu | 2,205,824 | 48.0 | 4913.5 |


## Dataset: amazon-toys-games

| Model | Seeds | Recall@20 | NDCG@20 | Recall@50 | NDCG@50 | Hit@20 | MRR@20 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| LightGCN | 1 | 0.1181 | 0.0567 | 0.1779 | 0.0688 | 0.1254 | 0.0407 |
| SGL | 1 | 0.1221 | 0.0603 | 0.1847 | 0.0730 | 0.1292 | 0.0440 |
| SGCL | 1 | 0.1322 | 0.0636 | 0.1951 | 0.0764 | 0.1400 | 0.0457 |
| FS-SGCL | 1 | 0.1364 | 0.0661 | 0.2001 | 0.0791 | 0.1442 | 0.0479 |

### Efficiency — amazon-toys-games

| Model | Device | Params | Best Epoch | Wall Time (s) |
| :--- | :--- | :--- | :--- | :--- |
| LightGCN | cuda | 2,005,632 | 148.0 | 3621.0 |
| SGL | cuda | 2,005,632 | 96.0 | 2865.9 |
| SGCL | cpu | 2,005,632 | 27.0 | 1385.3 |
| FS-SGCL | cpu | 2,005,632 | 37.0 | 3644.8 |

