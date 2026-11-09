|                          |   trainable params (M) |   avg train time (s, CPU) |   inference (examples/s, CPU) |
|:-------------------------|-----------------------:|--------------------------:|------------------------------:|
| BiLSTM+Att (no graph)    |                    1.5 |                     544.4 |                        1457.5 |
| GCN (dep. tree)          |                    2.2 |                    1192.1 |                         538.6 |
| GAT (dep. tree)          |                    2.2 |                     793.8 |                         670.4 |
| R-GAT (dep. tree + rel.) |                    2.3 |                     890.1 |                         682   |
| RoBERTa (fine-tuned)     |                  124.1 |                     981.2 |                          77.6 |
| RoBERTa+GCN              |                  124.6 |                     844.8 |                          75.8 |
| RoBERTa+R-GAT            |                  125   |                     858.6 |                          75.2 |