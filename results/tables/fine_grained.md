### Rest14: accuracy (%) by subgroup (seed-averaged)

| group                                             |    n |   BiLSTM+Att (no graph) |   GCN (dep. tree) |   GAT (dep. tree) |   R-GAT (dep. tree + rel.) |   RoBERTa (fine-tuned) |   RoBERTa+GCN |   RoBERTa+R-GAT |
|:--------------------------------------------------|-----:|------------------------:|------------------:|------------------:|---------------------------:|-----------------------:|--------------:|----------------:|
| all                                               | 1120 |                    79   |              79.1 |              78.5 |                       79.1 |                   84.9 |          84.5 |            84.1 |
| single-aspect sentence                            |  305 |                    76.4 |              77.2 |              77.7 |                       78.9 |                   86.2 |          85.6 |            85.6 |
| multi-aspect sentence                             |  815 |                    79.9 |              79.8 |              78.8 |                       79.2 |                   84.4 |          84   |            83.6 |
| mixed-polarity sentence                           |  218 |                    63.6 |              60.6 |              58.9 |                       61.5 |                   70.2 |          68.8 |            69.3 |
| contrast marker (but/however/...)                 |  156 |                    82.7 |              75.9 |              75.4 |                       78.6 |                   85.3 |          84.6 |            87.8 |
| length <= 15                                      |  493 |                    80.5 |              81.5 |              80.9 |                       80.6 |                   86.4 |          86.2 |            85.8 |
| length 16-30                                      |  500 |                    77.1 |              77.3 |              76.7 |                       78   |                   82.8 |          82.8 |            81.8 |
| length > 30                                       |  127 |                    80.6 |              76.6 |              76.4 |                       78   |                   87.4 |          84.3 |            86.6 |
| opinion word: tree dist 1                         |  253 |                    88.3 |              89.3 |              89.9 |                       88.4 |                   90.9 |          93.3 |            90.9 |
| opinion word: tree dist 2                         |  300 |                    88.7 |              88.8 |              89   |                       89.8 |                   92.3 |          92.7 |            92.3 |
| opinion word: tree dist >= 3                      |  347 |                    77.8 |              75.1 |              75.9 |                       76.5 |                   82.1 |          81.6 |            81.3 |
| linear dist >= 4                                  |  359 |                    76.9 |              75.3 |              76.2 |                       77.8 |                   82.7 |          82.7 |            82.5 |
| linear >> tree (lin - tree >= 3)                  |  215 |                    76   |              73   |              75   |                       76   |                   80.5 |          79.5 |            80   |
| distractor (lin-closest opinion not tree-closest) |   76 |                    79.8 |              72.4 |              74.1 |                       73.7 |                   82.9 |          80.3 |            80.3 |

### Lap14: accuracy (%) by subgroup (seed-averaged)

| group                                             |   n |   BiLSTM+Att (no graph) |   GCN (dep. tree) |   GAT (dep. tree) |   R-GAT (dep. tree + rel.) |   RoBERTa (fine-tuned) |   RoBERTa+GCN |   RoBERTa+R-GAT |
|:--------------------------------------------------|----:|------------------------:|------------------:|------------------:|---------------------------:|-----------------------:|--------------:|----------------:|
| all                                               | 638 |                    70.6 |              69.3 |              68.8 |                       70.5 |                   77.1 |          79   |            77.9 |
| single-aspect sentence                            | 278 |                    67.9 |              68.3 |              67.7 |                       68.3 |                   80.2 |          78.4 |            77.3 |
| multi-aspect sentence                             | 360 |                    72.7 |              70   |              69.5 |                       72.1 |                   74.7 |          79.4 |            78.3 |
| mixed-polarity sentence                           |  97 |                    58.4 |              51.9 |              46   |                       48.5 |                   42.3 |          56.7 |            58.8 |
| contrast marker (but/however/...)                 |  70 |                    55.2 |              53.3 |              49   |                       50.5 |                   58.6 |          60   |            61.4 |
| length <= 15                                      | 308 |                    75.4 |              73.6 |              75.9 |                       76.8 |                   85.4 |          86.4 |            84.1 |
| length 16-30                                      | 273 |                    66.5 |              66.1 |              63.4 |                       65.3 |                   70.3 |          71.4 |            72.2 |
| length > 30                                       |  57 |                    63.7 |              61.4 |              56.1 |                       60.8 |                   64.9 |          75.4 |            71.9 |
| opinion word: tree dist 1                         | 126 |                    83.6 |              81.5 |              81.2 |                       84.9 |                   84.9 |          86.5 |            85.7 |
| opinion word: tree dist 2                         | 133 |                    78.9 |              78.7 |              78.7 |                       81   |                   87.2 |          89.5 |            85.7 |
| opinion word: tree dist >= 3                      | 152 |                    61.8 |              61.4 |              59   |                       61   |                   70.4 |          69.7 |            71.1 |
| linear dist >= 4                                  | 169 |                    63.3 |              63.7 |              62.5 |                       65.7 |                   73.4 |          73.4 |            74   |
| linear >> tree (lin - tree >= 3)                  | 108 |                    66.4 |              64.2 |              61.7 |                       66.4 |                   73.1 |          74.1 |            73.1 |
| distractor (lin-closest opinion not tree-closest) |  28 |                    73.8 |              70.2 |              63.1 |                       64.3 |                   64.3 |          71.4 |            71.4 |

### MAMS: accuracy (%) by subgroup (seed-averaged)

| group                                             |    n |   BiLSTM+Att (no graph) |   GCN (dep. tree) |   GAT (dep. tree) |   R-GAT (dep. tree + rel.) |   RoBERTa (fine-tuned) |   RoBERTa+GCN |   RoBERTa+R-GAT |
|:--------------------------------------------------|-----:|------------------------:|------------------:|------------------:|---------------------------:|-----------------------:|--------------:|----------------:|
| all                                               | 1336 |                    78   |              78.4 |              77.9 |                       78.5 |                    nan |           nan |             nan |
| single-aspect sentence                            |    2 |                   100   |             100   |             100   |                      100   |                    nan |           nan |             nan |
| multi-aspect sentence                             | 1334 |                    77.9 |              78.4 |              77.9 |                       78.4 |                    nan |           nan |             nan |
| mixed-polarity sentence                           | 1334 |                    77.9 |              78.4 |              77.9 |                       78.4 |                    nan |           nan |             nan |
| contrast marker (but/however/...)                 |  347 |                    78.4 |              77.9 |              77   |                       78.8 |                    nan |           nan |             nan |
| length <= 15                                      |  127 |                    78.2 |              81.4 |              81.4 |                       80.3 |                    nan |           nan |             nan |
| length 16-30                                      |  731 |                    78   |              78.6 |              77.7 |                       78.3 |                    nan |           nan |             nan |
| length > 30                                       |  478 |                    77.8 |              77.5 |              77.3 |                       78.2 |                    nan |           nan |             nan |
| opinion word: tree dist 1                         |  165 |                    74.9 |              79.2 |              76.2 |                       76   |                    nan |           nan |             nan |
| opinion word: tree dist 2                         |  235 |                    74.5 |              77.3 |              76   |                       77.3 |                    nan |           nan |             nan |
| opinion word: tree dist >= 3                      |  597 |                    77.4 |              77.2 |              77   |                       77.1 |                    nan |           nan |             nan |
| linear dist >= 4                                  |  627 |                    75.6 |              76.5 |              75.5 |                       75.9 |                    nan |           nan |             nan |
| linear >> tree (lin - tree >= 3)                  |  441 |                    74.7 |              77.5 |              76.3 |                       76.3 |                    nan |           nan |             nan |
| distractor (lin-closest opinion not tree-closest) |   97 |                    78.4 |              77.7 |              79.7 |                       80.8 |                    nan |           nan |             nan |

### Yelp: accuracy (%) by subgroup (seed-averaged)

| group                                             |   n |   BiLSTM+Att (no graph) |   GCN (dep. tree) |   GAT (dep. tree) |   R-GAT (dep. tree + rel.) |   RoBERTa (fine-tuned) |   RoBERTa+GCN |   RoBERTa+R-GAT |
|:--------------------------------------------------|----:|------------------------:|------------------:|------------------:|---------------------------:|-----------------------:|--------------:|----------------:|
| all                                               | 245 |                    84.4 |              84.8 |              83.4 |                       84.5 |                   80.8 |          87.3 |            84.9 |
| single-aspect sentence                            | 136 |                    81.1 |              83.3 |              80.9 |                       81.6 |                   77.9 |          85.3 |            82.4 |
| multi-aspect sentence                             | 109 |                    88.4 |              86.5 |              86.5 |                       88.1 |                   84.4 |          89.9 |            88.1 |
| mixed-polarity sentence                           |  22 |                    77.3 |              62.1 |              62.1 |                       60.6 |                   54.5 |          63.6 |            59.1 |
| contrast marker (but/however/...)                 |  64 |                    74   |              72.9 |              69.8 |                       70.3 |                   57.8 |          79.7 |            70.3 |
| length <= 15                                      | 116 |                    88.2 |              90.5 |              89.1 |                       89.7 |                   84.5 |          87.1 |            86.2 |
| length 16-30                                      | 110 |                    80   |              79.1 |              78.5 |                       80   |                   75.5 |          88.2 |            83.6 |
| length > 30                                       |  19 |                    86   |              82.5 |              77.2 |                       78.9 |                   89.5 |          84.2 |            84.2 |
| opinion word: tree dist 1                         |  58 |                    93.7 |              89.7 |              90.8 |                       90.2 |                   93.1 |          98.3 |            96.6 |
| opinion word: tree dist 2                         |  89 |                    88.4 |              91   |              90.6 |                       90.3 |                   82   |          88.8 |            84.3 |
| opinion word: tree dist >= 3                      |  71 |                    75.1 |              74.6 |              72.3 |                       75.1 |                   76.1 |          83.1 |            81.7 |
| linear dist >= 4                                  |  79 |                    71.7 |              73.4 |              70.5 |                       70.9 |                   77.2 |          79.7 |            78.5 |
| linear >> tree (lin - tree >= 3)                  |  48 |                    70.1 |              71.5 |              67.4 |                       71.5 |                   75   |          79.2 |            79.2 |
| distractor (lin-closest opinion not tree-closest) |  17 |                    80.4 |              82.4 |              78.4 |                       78.4 |                   88.2 |         100   |           100   |

### Amazon23 (distant): accuracy (%) by subgroup (seed-averaged)

| group                                             |   n |   BiLSTM+Att (no graph) |   GCN (dep. tree) |   GAT (dep. tree) |   R-GAT (dep. tree + rel.) |   RoBERTa (fine-tuned) |   RoBERTa+GCN |   RoBERTa+R-GAT |
|:--------------------------------------------------|----:|------------------------:|------------------:|------------------:|---------------------------:|-----------------------:|--------------:|----------------:|
| all                                               | 858 |                    93.4 |              94.1 |              95.3 |                       94.4 |                   95.7 |          96.3 |            95.5 |
| single-aspect sentence                            | 610 |                    94   |              94.8 |              95.4 |                       95   |                   95.6 |          95.7 |            95.2 |
| multi-aspect sentence                             | 248 |                    91.8 |              92.3 |              94.9 |                       93.1 |                   96   |          97.6 |            96   |
| mixed-polarity sentence                           |   0 |                   nan   |             nan   |             nan   |                      nan   |                  nan   |         nan   |           nan   |
| contrast marker (but/however/...)                 |   0 |                   nan   |             nan   |             nan   |                      nan   |                  nan   |         nan   |           nan   |
| length <= 15                                      | 390 |                    96.4 |              96.9 |              97.5 |                       96.8 |                   97.9 |          97.4 |            97.7 |
| length 16-30                                      | 366 |                    90.4 |              91.7 |              93.4 |                       92.2 |                   93.7 |          95.4 |            93.7 |
| length > 30                                       | 102 |                    92.2 |              91.8 |              93.5 |                       93.5 |                   94.1 |          95.1 |            93.1 |
| opinion word: tree dist 1                         | 183 |                    94.7 |              95.4 |              96.2 |                       94.4 |                   96.7 |          96.7 |            97.3 |
| opinion word: tree dist 2                         | 167 |                    95.8 |              95.8 |              96.6 |                       97.2 |                   97.6 |          98.2 |            98.2 |
| opinion word: tree dist >= 3                      | 315 |                    92.1 |              93.8 |              94.8 |                       94.6 |                   96.2 |          96.2 |            95.2 |
| linear dist >= 4                                  | 323 |                    92   |              93.2 |              94.4 |                       93.9 |                   95.4 |          96   |            94.7 |
| linear >> tree (lin - tree >= 3)                  | 192 |                    91   |              91.1 |              93.8 |                       92.2 |                   95.3 |          96.4 |            94.3 |
| distractor (lin-closest opinion not tree-closest) |  65 |                    91.3 |              91.8 |              92.8 |                       91.8 |                   92.3 |          95.4 |            95.4 |

### Amazon23 (gold): accuracy (%) by subgroup (seed-averaged)

| group                                             |   n |   BiLSTM+Att (no graph) |   GCN (dep. tree) |   GAT (dep. tree) |   R-GAT (dep. tree + rel.) |   RoBERTa (fine-tuned) |   RoBERTa+GCN |   RoBERTa+R-GAT |
|:--------------------------------------------------|----:|------------------------:|------------------:|------------------:|---------------------------:|-----------------------:|--------------:|----------------:|
| all                                               | 225 |                    49.5 |              51.1 |              49.2 |                       49.8 |                   51.1 |          54.2 |            50.7 |
| single-aspect sentence                            |  82 |                    58.1 |              63   |              59.8 |                       57.7 |                   62.2 |          67.1 |            62.2 |
| multi-aspect sentence                             | 143 |                    44.5 |              44.3 |              43.1 |                       45.2 |                   44.8 |          46.9 |            44.1 |
| mixed-polarity sentence                           |  93 |                    46.2 |              45.9 |              45.2 |                       47   |                   46.2 |          47.3 |            45.2 |
| contrast marker (but/however/...)                 | 141 |                    42.8 |              43.3 |              42.1 |                       43.7 |                   44.7 |          47.5 |            44   |
| length <= 15                                      |  48 |                    61.8 |              63.2 |              61.8 |                       62.5 |                   60.4 |          68.8 |            60.4 |
| length 16-30                                      | 126 |                    47.6 |              49.5 |              47.1 |                       47.9 |                   51.6 |          53.2 |            51.6 |
| length > 30                                       |  51 |                    42.5 |              43.8 |              42.5 |                       42.5 |                   41.2 |          43.1 |            39.2 |
| opinion word: tree dist 1                         |  26 |                    73.1 |              65.4 |              69.2 |                       70.5 |                   73.1 |          73.1 |            69.2 |
| opinion word: tree dist 2                         |  27 |                    54.3 |              54.3 |              54.3 |                       53.1 |                   59.3 |          59.3 |            63   |
| opinion word: tree dist >= 3                      |  96 |                    42.7 |              44.8 |              43.1 |                       42.7 |                   43.8 |          44.8 |            43.8 |
| linear dist >= 4                                  | 100 |                    43   |              44   |              43.7 |                       43.3 |                   45   |          46   |            44   |
| linear >> tree (lin - tree >= 3)                  |  72 |                    42.6 |              42.6 |              42.6 |                       43.5 |                   44.4 |          45.8 |            43.1 |
| distractor (lin-closest opinion not tree-closest) |   8 |                    62.5 |              66.7 |              62.5 |                       62.5 |                   62.5 |          62.5 |            62.5 |

