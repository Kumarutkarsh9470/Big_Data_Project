| dataset                                         | A                        | B                        |   acc(A)-acc(B) |   p (A<=B) |
|:------------------------------------------------|:-------------------------|:-------------------------|----------------:|-----------:|
| Rest14                                          | GCN (dep. tree)          | BiLSTM+Att (no graph)    |            0.12 |     0.4384 |
| Rest14                                          | GAT (dep. tree)          | BiLSTM+Att (no graph)    |           -0.45 |     0.738  |
| Rest14                                          | R-GAT (dep. tree + rel.) | BiLSTM+Att (no graph)    |            0.18 |     0.4026 |
| Rest14                                          | R-GAT (dep. tree + rel.) | GCN (dep. tree)          |            0.06 |     0.4676 |
| Rest14                                          | GAT (dep. tree)          | GCN (dep. tree)          |           -0.57 |     0.8202 |
| Rest14                                          | RoBERTa (fine-tuned)     | R-GAT (dep. tree + rel.) |            4.88 |     0      |
| Rest14                                          | RoBERTa+R-GAT            | RoBERTa (fine-tuned)     |            0.09 |     0.4508 |
| Rest14                                          | RoBERTa+GCN              | RoBERTa (fine-tuned)     |            0.4  |     0.2724 |
| Lap14                                           | GCN (dep. tree)          | BiLSTM+Att (no graph)    |           -1.31 |     0.8582 |
| Lap14                                           | GAT (dep. tree)          | BiLSTM+Att (no graph)    |           -1.83 |     0.941  |
| Lap14                                           | R-GAT (dep. tree + rel.) | BiLSTM+Att (no graph)    |           -0.1  |     0.5324 |
| Lap14                                           | R-GAT (dep. tree + rel.) | GCN (dep. tree)          |            1.2  |     0.0782 |
| Lap14                                           | GAT (dep. tree)          | GCN (dep. tree)          |           -0.52 |     0.7146 |
| Lap14                                           | RoBERTa (fine-tuned)     | R-GAT (dep. tree + rel.) |            7.18 |     0      |
| Lap14                                           | RoBERTa+R-GAT            | RoBERTa (fine-tuned)     |            0.47 |     0.3596 |
| Lap14                                           | RoBERTa+GCN              | RoBERTa (fine-tuned)     |            1.02 |     0.184  |
| MAMS                                            | GCN (dep. tree)          | BiLSTM+Att (no graph)    |            0.47 |     0.2608 |
| MAMS                                            | GAT (dep. tree)          | BiLSTM+Att (no graph)    |           -0.05 |     0.5406 |
| MAMS                                            | R-GAT (dep. tree + rel.) | BiLSTM+Att (no graph)    |            0.5  |     0.2516 |
| MAMS                                            | R-GAT (dep. tree + rel.) | GCN (dep. tree)          |            0.02 |     0.495  |
| MAMS                                            | GAT (dep. tree)          | GCN (dep. tree)          |           -0.52 |     0.8438 |
| MAMS                                            | RoBERTa (fine-tuned)     | R-GAT (dep. tree + rel.) |            4.99 |     0      |
| MAMS                                            | RoBERTa+R-GAT            | RoBERTa (fine-tuned)     |           -0.22 |     0.6364 |
| Yelp                                            | GCN (dep. tree)          | BiLSTM+Att (no graph)    |            0.41 |     0.4144 |
| Yelp                                            | GAT (dep. tree)          | BiLSTM+Att (no graph)    |           -0.95 |     0.7368 |
| Yelp                                            | R-GAT (dep. tree + rel.) | BiLSTM+Att (no graph)    |            0.14 |     0.4652 |
| Yelp                                            | R-GAT (dep. tree + rel.) | GCN (dep. tree)          |           -0.27 |     0.6066 |
| Yelp                                            | GAT (dep. tree)          | GCN (dep. tree)          |           -1.36 |     0.9038 |
| Yelp                                            | RoBERTa (fine-tuned)     | R-GAT (dep. tree + rel.) |           -3.47 |     0.9766 |
| Yelp                                            | RoBERTa+R-GAT            | RoBERTa (fine-tuned)     |            2.65 |     0.0266 |
| Yelp                                            | RoBERTa+GCN              | RoBERTa (fine-tuned)     |            3.67 |     0.0002 |
| Amazon23 (distant)                              | GCN (dep. tree)          | BiLSTM+Att (no graph)    |            0.74 |     0.0574 |
| Amazon23 (distant)                              | GAT (dep. tree)          | BiLSTM+Att (no graph)    |            1.9  |     0      |
| Amazon23 (distant)                              | R-GAT (dep. tree + rel.) | BiLSTM+Att (no graph)    |            1.09 |     0.0158 |
| Amazon23 (distant)                              | R-GAT (dep. tree + rel.) | GCN (dep. tree)          |            0.35 |     0.1786 |
| Amazon23 (distant)                              | GAT (dep. tree)          | GCN (dep. tree)          |            1.17 |     0.0022 |
| Amazon23 (distant)                              | RoBERTa (fine-tuned)     | R-GAT (dep. tree + rel.) |            1.24 |     0.0382 |
| Amazon23 (distant)                              | RoBERTa+R-GAT            | RoBERTa (fine-tuned)     |           -0.23 |     0.7206 |
| Amazon23 (distant)                              | RoBERTa+GCN              | RoBERTa (fine-tuned)     |            0.58 |     0.1324 |
| Amazon23 (gold) ← trained on Amazon23 (distant) | GCN (dep. tree)          | BiLSTM+Att (no graph)    |            1.63 |     0.125  |
| Amazon23 (gold) ← trained on Amazon23 (distant) | GAT (dep. tree)          | BiLSTM+Att (no graph)    |           -0.3  |     0.618  |
| Amazon23 (gold) ← trained on Amazon23 (distant) | R-GAT (dep. tree + rel.) | BiLSTM+Att (no graph)    |            0.3  |     0.4258 |
| Amazon23 (gold) ← trained on Amazon23 (distant) | R-GAT (dep. tree + rel.) | GCN (dep. tree)          |           -1.33 |     0.8858 |
| Amazon23 (gold) ← trained on Amazon23 (distant) | GAT (dep. tree)          | GCN (dep. tree)          |           -1.93 |     0.9708 |
| Amazon23 (gold) ← trained on Amazon23 (distant) | RoBERTa (fine-tuned)     | R-GAT (dep. tree + rel.) |            1.33 |     0.2566 |
| Amazon23 (gold) ← trained on Amazon23 (distant) | RoBERTa+R-GAT            | RoBERTa (fine-tuned)     |           -0.44 |     0.7582 |
| Amazon23 (gold) ← trained on Amazon23 (distant) | RoBERTa+GCN              | RoBERTa (fine-tuned)     |            3.11 |     0.0004 |
| Amazon23 (gold) ← trained on Lap14              | GCN (dep. tree)          | BiLSTM+Att (no graph)    |           -3.26 |     0.9322 |
| Amazon23 (gold) ← trained on Lap14              | GAT (dep. tree)          | BiLSTM+Att (no graph)    |           -7.41 |     1      |
| Amazon23 (gold) ← trained on Lap14              | R-GAT (dep. tree + rel.) | BiLSTM+Att (no graph)    |           -5.19 |     0.9944 |
| Amazon23 (gold) ← trained on Lap14              | R-GAT (dep. tree + rel.) | GCN (dep. tree)          |           -1.93 |     0.8912 |
| Amazon23 (gold) ← trained on Lap14              | GAT (dep. tree)          | GCN (dep. tree)          |           -4.15 |     0.9988 |
| Amazon23 (gold) ← trained on Lap14              | RoBERTa (fine-tuned)     | R-GAT (dep. tree + rel.) |           13.04 |     0      |
| Amazon23 (gold) ← trained on Lap14              | RoBERTa+R-GAT            | RoBERTa (fine-tuned)     |            4    |     0.0748 |
| Amazon23 (gold) ← trained on Lap14              | RoBERTa+GCN              | RoBERTa (fine-tuned)     |            5.11 |     0.0194 |