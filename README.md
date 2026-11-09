# Dependency-Tree GNNs vs. Transformers for Aspect-Based Sentiment Analysis

Course project (Big Data). We compare graph neural networks that run over dependency parses (GCN, GAT,
relation-aware GAT) with a fine-tuned RoBERTa baseline and with RoBERTa + GNN hybrids for aspect-based
sentiment classification. Evaluation covers Amazon Reviews 2023 and Yelp, plus SemEval-2014 and MAMS
for comparison with prior work. All experiments run on CPU.

| Deliverable | Location |
|---|---|
| Report (ACL format, 9 pages) | [`report/report.pdf`](report/report.pdf) |
| Slides | [`slides/ABSA_GNN_vs_Transformer_slides.pptx`](slides/ABSA_GNN_vs_Transformer_slides.pptx) |
| Executed notebook | [`notebooks/ABSA_GNN_vs_Transformer.ipynb`](notebooks/ABSA_GNN_vs_Transformer.ipynb) |
| Result tables and figures | [`results/tables/`](results/tables), [`results/figures/`](results/figures) |
| Raw runs (config, learning curve, metrics, per-example predictions) | [`results/runs/`](results/runs) |

## Contents

1. [Task](#1-task)
2. [Key findings](#2-key-findings)
3. [Results](#3-results)
4. [Data](#4-data)
5. [Models and training protocol](#5-models-and-training-protocol)
6. [Repository structure](#6-repository-structure)
7. [Reproducing the results](#7-reproducing-the-results)
8. [Limitations](#8-limitations)
9. [References](#9-references)

## 1. Task

Given a review sentence and an aspect term in it, predict the sentiment expressed towards that aspect
(positive, negative or neutral). In *"The battery life is great but the screen is too dim"* the aspect
*battery life* is positive and *screen* is negative. In the dependency parse, *dim* is two arcs from
*screen*, while in the word sequence *great* and *dim* are equally close to it. Graph models over the
parse are meant to exploit this.

## 2. Key findings

* **The encoder matters more than the graph.** Fine-tuned RoBERTa is 5 to 8 accuracy points better than
  every GloVe-based model on Rest14, Lap14 and MAMS.
* **On a GloVe BiLSTM, the parse adds no reliable gain.** GCN, GAT and R-GAT are within 1.8 points of the
  BiLSTM baseline (no significant difference). A random tree of the same size is clearly worse than the
  parse on Rest14, but a simple word-chain graph does as well as the parse.
* **On RoBERTa, the parse helps where aspects disagree.** On sentences whose aspects carry different
  polarities, adding a dependency GNN raises accuracy from 44 to 60-62 % on Lap14 and from 47 to 59-62 % on
  the Amazon gold set, in both seeds. The effect disappears on MAMS, whose 11k training sentences are all of
  this type.
* **Star-rating labels are a trap.** Aspect labels derived from ratings are about 70 % correct. Models
  trained on them score 93-96 % on a test set labelled the same way but only 49-54 % on manually labelled
  Amazon data; a model trained on human-labelled laptop reviews reaches 59-64 %.
* **Throughput.** Over 97k Amazon reviews (588k aspect mentions) the GloVe R-GAT processes about 2,200
  mentions/s on 4 CPU cores; RoBERTa+GCN processes 77 mentions/s but is 15 points more accurate on the
  Amazon gold set.

## 3. Results

### 3.1 Main comparison

Test accuracy / macro-F1 (%). GloVe models: mean of 3 seeds. RoBERTa models: mean of 2 seeds on Rest14,
Lap14, Yelp and Lap14 to Amazon23 gold; 1 seed on MAMS. Model selection always uses the dev split.

| Model | Rest14 | Lap14 | MAMS | Yelp | Lap14 to Amazon23 gold |
|---|:---:|:---:|:---:|:---:|:---:|
| BiLSTM+Att (no graph) | 78.96 / 68.38 | 70.59 / 64.48 | 77.97 / 76.88 | 84.35 / 61.05 | 51.26 / 50.36 |
| GCN | 79.08 / 69.99 | 69.28 / 64.26 | 78.44 / 77.39 | 84.76 / 64.53 | 48.00 / 48.08 |
| GAT | 78.51 / 67.36 | 68.76 / 62.28 | 77.92 / 77.08 | 83.40 / 61.98 | 43.85 / 43.68 |
| R-GAT | 79.14 / 69.13 | 70.48 / 64.78 | 78.47 / 77.38 | 84.49 / 63.87 | 46.07 / 46.19 |
| RoBERTa | 84.02 / 74.69 | 77.66 / 72.76 | **83.46 / 83.08** | 81.02 / 51.92 | 59.11 / 57.32 |
| RoBERTa+GCN | **84.42 / 75.98** | **78.68 / 74.20** | 83.08 / 82.68 | **84.69** / 55.18 | **64.22 / 63.50** |
| RoBERTa+R-GAT | 84.11 / 75.57 | 78.13 / 73.94 | 83.23 / 82.83 | 83.67 / **57.64** | 63.11 / 61.86 |

Seed standard deviations are 0.4 to 2.4 accuracy points. Full tables with standard deviations:
[`results/tables/main_results.md`](results/tables/main_results.md); significance tests:
[`results/tables/significance.md`](results/tables/significance.md).

### 3.2 Graph-structure ablation (GloVe encoder, accuracy / macro-F1, 3 seeds)

| Graph | Rest14 GCN | Rest14 GAT | Lap14 GCN | Lap14 GAT |
|---|:---:|:---:|:---:|:---:|
| Dependency tree | 79.1 / 70.0 | 78.5 / 67.4 | 69.3 / 64.3 | 68.8 / 62.3 |
| Word chain | 80.3 / 70.9 | 79.9 / 69.5 | 69.6 / 63.7 | 70.5 / 65.1 |
| Random tree | 76.6 / 64.8 | 76.6 / 64.8 | 68.5 / 62.7 | 68.0 / 62.3 |
| Complete graph | 77.9 / 67.7 | 78.5 / 67.5 | 68.6 / 62.3 | 68.7 / 62.4 |
| No graph (BiLSTM) | 79.0 / 68.4 | | 70.6 / 64.5 | |

1 to 4 GCN layers change accuracy by at most 1.3 points ([`results/tables/ablation.md`](results/tables/ablation.md)).

### 3.3 Sentences whose aspects have different polarities (accuracy %, mean over seeds)

| Test set | n | RoBERTa | RoBERTa+GCN | RoBERTa+R-GAT |
|---|:---:|:---:|:---:|:---:|
| Lap14 | 97 | 44.3 | 59.8 | 61.9 |
| Lap14 to Amazon23 gold | 93 | 46.8 | 59.1 | 62.4 |
| Rest14 | 218 | 67.2 | 68.8 | 69.7 |
| MAMS | 1,334 | 83.4 | 83.1 | 83.2 |

More breakdowns (contrast markers, tree and linear distance to the opinion word):
[`results/tables/fine_grained.md`](results/tables/fine_grained.md).

### 3.4 Distant supervision and transfer (accuracy %)

| Train to test | BiLSTM | R-GAT | RoBERTa | RoBERTa+GCN |
|---|:---:|:---:|:---:|:---:|
| Amazon23 distant to Amazon23 distant | 93.4 | 94.4 | 95.7 | 96.3 |
| Amazon23 distant to Amazon23 gold | 49.5 | 49.8 | 51.1 | 54.2 |
| Lap14 to Amazon23 gold | 51.3 | 46.1 | 59.1 | 64.2 |
| Rest14 to Yelp | 84.9 | 85.6 | 88.0 | 86.9 |
| Yelp to Rest14 | 70.3 | 72.4 | 75.3 | 75.5 |

### 3.5 Corpus-scale run on Amazon Reviews 2023 (4 CPU cores)

| | GloVe R-GAT | RoBERTa+GCN |
|---|:---:|:---:|
| Reviews processed | 97,487 | 10,000 |
| Aspect mentions | 588,417 | 65,432 |
| Inference throughput (mentions/s) | 2,207 | 77 |
| End-to-end throughput incl. parsing (reviews/s) | 223 | 11 |
| Accuracy on Amazon23 gold | 48.9 % | 64.2 % |
| Spearman correlation with star rating (same 10k reviews) | 0.39 | 0.48 |

Outputs: [`results/amazon_mining/`](results/amazon_mining) (per-aspect summary, figures) and
[`results/amazon_mining_roberta_gcn_10k/`](results/amazon_mining_roberta_gcn_10k).

## 4. Data

| Dataset | Domain | Labels | Train | Dev | Test |
|---|---|---|---:|---:|---:|
| Rest14 | restaurants | SemEval-2014 Task 4 | 3,246 | 362 | 1,120 |
| Lap14 | laptops | SemEval-2014 Task 4 | 2,095 | 233 | 638 |
| MAMS | restaurants | human (every sentence has 2+ aspects with different polarities) | 11,186 | 1,332 | 1,336 |
| Yelp | Yelp reviews | human (PyABSA collection) | 727 | 81 | 245 |
| Amazon23 distant | Amazon Reviews 2023, Electronics | star rating + VADER | 6,805 | 837 | 858 |
| Amazon23 gold | Amazon Reviews 2023, Electronics | manual | | | 225 |

**Preprocessing.** All texts are tokenised and parsed with spaCy `en_core_web_lg` 3.8. Long reviews are
cropped to the sentence containing the aspect. Dev splits for Rest14, Lap14 and Yelp are 10 % of the
training sentences, split by sentence so that aspects of one sentence stay together.

**Amazon23 distant.** 40k reviews are parsed and an aspect lexicon is mined (400 frequent nouns and 200
compound nouns such as *battery life*). A sentence's aspects are labelled positive if the review has 4-5
stars and VADER >= 0.5, negative if it has 1-2 stars and VADER <= -0.3, and neutral if the sentence contains
no sentiment word. Sentences with contrast markers are removed. A manual audit of 100 labels
(`data/annotations/amazon23_distant_audit.tsv`) found 11 % invalid aspects and 70 % correct labels on the rest.

**Amazon23 gold.** 225 aspect mentions from 3k held-out reviews (no overlap with the distant set), two
thirds from multi-aspect sentences with a contrast marker, labelled per aspect following the SemEval-2014
guidelines (`data/annotations/amazon23_gold_labels.tsv`).

**Sources.** The experiment machine could not reach the HuggingFace Hub or the official dataset hosts,
so `scripts/setup_resources.sh` downloads everything from GitHub at pinned commits: a 100k-review
Electronics subset of Amazon Reviews 2023 (Hou et al., 2024), the PyABSA dataset collection (SemEval-2014,
MAMS, Yelp), GloVe 6B 300d and the spaCy models.

## 5. Models and training protocol

| Model | Encoder | Graph layer (2 layers) | Readout |
|---|---|---|---|
| BiLSTM+Att | frozen GloVe 300d + POS, BiLSTM 2x300 | none | aspect-query attention |
| GCN | same | GCN over the parse, position-weighted (ASGCN) | same |
| GAT | same | 4-head graph attention over parse neighbours | same |
| R-GAT | same | GAT with dependency-relation embeddings | same |
| RoBERTa | RoBERTa-base, fine-tuned | none | `<s>` token |
| RoBERTa+GCN | same | GCN over word states | `<s>` + mean of aspect nodes |
| RoBERTa+R-GAT | same | R-GAT over word states | `<s>` + mean of aspect nodes |

* **GloVe models:** Adam, lr 1e-3, batch 32, up to 30 epochs, early stopping (patience 8), 3 seeds.
* **RoBERTa models:** AdamW, lr 2e-5 (encoder) / 1e-3 (new layers), batch 16, 4 epochs (3 on MAMS),
  10 % warm-up, 1-2 seeds.
* **Selection:** the checkpoint with the best dev accuracy + macro-F1 is evaluated once on test.
* **Significance:** paired bootstrap over test examples (5,000 resamples).
* **RoBERTa weights:** extracted from spaCy `en_core_web_trf` (`scripts/export_roberta_from_spacy.py`);
  with Hub access use `--plm roberta-base`.

## 6. Repository structure

```
.
├── src/absa/                 core package
│   ├── data.py               readers, sentence cropping, spaCy parsing
│   ├── graph.py              adjacency matrices (dependency, chain, random tree, complete)
│   ├── models.py             BiLSTM, GCN, GAT, R-GAT, RoBERTa and hybrids
│   └── train.py              training / evaluation CLI
├── scripts/
│   ├── setup_resources.sh    download parsers, GloVe, datasets, RoBERTa
│   ├── prepare_gold.py       build Rest14 / Lap14 / Yelp / MAMS splits
│   ├── build_amazon2023_absa.py   distant-supervision set + annotation pool
│   ├── make_amazon_gold.py   build the gold test set, score the audit
│   ├── run_grid.py, run_all.sh    full experiment schedule (resumable)
│   ├── analyze.py            tables, significance tests, figures
│   ├── mine_amazon_aspects.py     corpus-scale inference
│   └── make_notebook.py      regenerate the notebook
├── data/
│   ├── processed/            train / dev / test jsonl with parses (ready to use)
│   └── annotations/          manual labels (gold set, audit)
├── results/                  runs/, tables/, figures/, amazon_mining*/
├── notebooks/                executed walkthrough notebook
├── report/                   LaTeX source and PDF (ACL template)
└── slides/                   presentation and its generator
```

## 7. Reproducing the results

**Requirements:** Python 3.10+, about 4 GB RAM, 2 GB disk for resources. A GPU is optional.

1. Install dependencies.
   ```bash
   pip install -r requirements.txt
   ```
2. Download resources into `./resources` (parsers, GloVe, raw datasets, RoBERTa).
   ```bash
   bash scripts/setup_resources.sh
   ```
3. *(Optional, the processed data is committed.)* Rebuild the datasets.
   ```bash
   python scripts/prepare_gold.py --datasets rest14 lap14 yelp mams
   ```
   ```bash
   python scripts/build_amazon2023_absa.py && python scripts/make_amazon_gold.py
   ```
4. Run all experiments (about 12 h on 4 CPU cores; finished runs are skipped on restart).
   ```bash
   bash scripts/run_all.sh
   ```
5. Build tables and figures.
   ```bash
   python scripts/analyze.py
   ```
6. Corpus-scale run (requires a saved model, see below).
   ```bash
   python scripts/mine_amazon_aspects.py --model-path results/models/lap14-amazon23_rgat.pt
   ```

**Single runs** (from `src/`):

```bash
python -m absa.train --dataset rest14 --model gcn --seed 1
```
```bash
python -m absa.train --dataset lap14 --model roberta-gcn --eval-on amazon23_gold
```
```bash
python -m absa.train --dataset lap14+amazon23 --model rgat --save-model ../results/models/lap14-amazon23_rgat.pt
```

| Option | Values |
|---|---|
| `--dataset` | `rest14`, `lap14`, `mams`, `yelp`, `amazon23`, or a union such as `lap14+amazon23` |
| `--model` | `lstm`, `gcn`, `gat`, `rgat`, `roberta`, `roberta-gcn`, `roberta-gat` |
| `--graph` | `dep` (default), `seq`, `random`, `full` |
| `--layers`, `--seed`, `--epochs` | graph depth, random seed, epoch budget |
| `--eval-on` | extra test sets, e.g. `amazon23_gold yelp` |
| `--save-model` | path to store the trained model |

Each run writes `results/runs/<name>.json` (configuration, learning curve, test metrics, timing) and
`<name>.preds.json` (per-example predictions and probabilities).

## 8. Limitations

* CPU-only budget: RoBERTa models use 1-2 seeds and no hyper-parameter search.
* RoBERTa weights come from spaCy `en_core_web_trf` and were further trained by spaCy on OntoNotes.
* Amazon data is a 100k-review sample of one category from a public mirror.
* The Amazon gold set and the label audit were annotated in a single pass with LLM assistance, without an
  inter-annotator agreement study. They should be checked by additional annotators.
* The Yelp test set is small (245 aspects, 16 neutral), so Yelp results have wide confidence intervals.

## 9. References

* Pontiki et al., 2014. SemEval-2014 Task 4: Aspect Based Sentiment Analysis.
* Jiang et al., 2019. A Challenge Dataset and Effective Models for Aspect-Based Sentiment Analysis (MAMS).
* Zhang et al., 2019. Aspect-based Sentiment Classification with Aspect-specific Graph Convolutional Networks.
* Wang et al., 2020. Relational Graph Attention Network for Aspect-based Sentiment Analysis.
* Dai et al., 2021. Does syntax matter? A strong baseline for Aspect-based Sentiment Analysis with RoBERTa.
* Liu et al., 2019. RoBERTa: A Robustly Optimized BERT Pretraining Approach.
* Hou et al., 2024. Bridging Language and Items for Retrieval and Recommendation (Amazon Reviews 2023).
* Yang et al., 2023. PyABSA: A Modularized Framework for Reproducible Aspect-based Sentiment Analysis.
