# Dependency-tree GNNs for Aspect-Based Sentiment Analysis
**GCN vs. GAT vs. a Transformer baseline on Amazon Reviews 2023 and Yelp**

Aspect-based sentiment classification (ABSC): given a review sentence and an aspect term in it
(*"The battery life is great but the **screen** is too dim"*), predict the polarity expressed towards
that aspect (positive / negative / neutral). We compare graph neural networks that run over the
sentence's **dependency tree** (GCN, GAT, relation-aware GAT) with a fine-tuned **Transformer**
(RoBERTa-base) and with hybrids (RoBERTa + dependency GNN). Everything runs on a CPU.

* Executable walkthrough with all results: [`notebooks/ABSA_GNN_vs_Transformer.ipynb`](notebooks/ABSA_GNN_vs_Transformer.ipynb)
* Result tables: [`results/tables/`](results/tables) · figures: [`results/figures/`](results/figures) · per-run JSON + per-example predictions: [`results/runs/`](results/runs)

## Key results
Test accuracy / macro-F1 (%), mean ± std over seeds (GloVe models: 3 seeds; RoBERTa models: 2 seeds on
Rest14/Lap14/Yelp, 1 seed on MAMS/Amazon23). Model selection always on a held-out dev split.
Full tables: [`results/tables/main_results.md`](results/tables/main_results.md).

| model | Rest14 | Lap14 | MAMS | Yelp | Lap14 → **Amazon23 gold** |
|---|---|---|---|---|---|
| BiLSTM+Att (no graph) | 78.96 / 68.38 | 70.59 / 64.48 | 77.97 / 76.88 | 84.35 / 61.05 | 51.26 / 50.36 |
| GCN (dep. tree) | 79.08 / 69.99 | 69.28 / 64.26 | 78.44 / 77.39 | 84.76 / 64.53 | 48.00 / 48.08 |
| GAT (dep. tree) | 78.51 / 67.36 | 68.76 / 62.28 | 77.92 / 77.08 | 83.40 / 61.98 | 43.85 / 43.68 |
| R-GAT (dep. tree + relations) | 79.14 / 69.13 | 70.48 / 64.78 | 78.47 / 77.38 | 84.49 / 63.87 | 46.07 / 46.19 |
| **RoBERTa** (fine-tuned) | 84.02 / 74.69 | 77.66 / 72.76 | **83.46 / 83.08** | 81.02 / 51.92 | 59.11 / 57.32 |
| RoBERTa + GCN | **84.42 / 75.98** | **78.68 / 74.20** | – | **84.69** / 55.18 | **64.22 / 63.50** |
| RoBERTa + R-GAT | 84.11 / 75.57 | 78.13 / 73.94 | 83.23 / 82.83 | 83.67 / **57.64** | 63.11 / 61.86 |

**Findings**
1. **The Transformer dominates the GloVe GNNs**: +5 to +8 accuracy on Rest14 / Lap14 / MAMS, and +8 to +20 on the
   manually labelled Amazon 2023 set.
2. **With static embeddings, the dependency graph adds little.** GCN/GAT/R-GAT are within seed noise of the
   BiLSTM in-domain. But the **syntax is not irrelevant**: a *random tree* with the same number of edges is clearly
   worse (Rest14 GCN 76.6 vs 79.1 acc, 64.8 vs 70.0 F1). A simple *sequential chain* graph does just as well
   as the dependency tree on top of a BiLSTM ([`ablation.md`](results/tables/ablation.md)). 1–4 GCN layers make no difference.
   Dependency GNNs transfer better from Yelp to Rest14 (+2.1–2.7 acc over the BiLSTM).
3. **Syntax helps the Transformer on hard, under-represented cases.** RoBERTa + dependency GNN beats RoBERTa on
   sentences whose aspects have *different* polarities ([`fine_grained.md`](results/tables/fine_grained.md)):
   Lap14 44.3 → 59.8 / 61.9, Amazon gold 46.8 → 59.1 / 62.4, Rest14 67.2 → 68.8 / 69.7, Yelp 54.5 → 61.4 / 63.6.
   On Amazon gold the overall gain is +5.1 (GCN, p = 0.02, paired bootstrap) and +4.0 (R-GAT, p = 0.07).
   On **MAMS**, where *every* training sentence is multi-aspect / mixed-polarity (11k), the gain disappears
   (83.2 vs 83.5): with enough hard training data the Transformer learns the aspect–opinion links on its own.
4. **Distant supervision from star ratings is a trap.** Models trained on distant Amazon labels score 93–96 % on
   the distant test set but only 49–54 % on the gold set: they learn the labelling heuristic. Our audit shows the
   labels are only ~70 % correct. Human-labelled *out-of-domain* data (Lap14) transfers much better (59–66 %).
   Training on Lap14 + distant Amazon jointly does not help either (GloVe R-GAT: 48.9 %, `results/runs_extra/`).
   Likewise Rest14 → Yelp (87–89 % for RoBERTa models) beats training on the small Yelp set itself (81–85 %).
5. **Cost at corpus scale** ([`results/amazon_mining/`](results/amazon_mining)): on 4 CPU cores the GloVe R-GAT
   processes 97k Amazon reviews → 500k sentences → **588k aspect mentions in 7.3 min** (2,207 mentions/s inference,
   223 reviews/s including parsing). RoBERTa+GCN runs at 77 mentions/s (**28× slower**). It is far more accurate on
   gold (64 vs 49 %) and its review-level aspect sentiment correlates better with star ratings
   (Spearman ρ = 0.48 vs 0.41) even though it never saw ratings. The two models agree on 62 % of mentions.
   Most negative aspects in Electronics: *battery*, *drive*, *unit*, *power*; most positive: *price*, *quality*, *sound*, *size*.

Efficiency (CPU, 4 threads): GloVe GNNs have 2.2 M trainable parameters and infer 540–680 examples/s;
RoBERTa models have 125 M parameters and infer ~72 examples/s ([`efficiency.md`](results/tables/efficiency.md)).

**Caveats.** The Amazon gold set (225 aspects) was annotated by the project's AI assistant following SemEval
guidelines and should be spot-checked by a team member. RoBERTa results use 1–2 seeds, and the paired bootstrap
captures example-level, not seed-level, variance. Our numbers come from dev-based model selection, so they are lower
than many papers that report the best test epoch (e.g. ASGCN reports 80.8 on Rest14; our GCN gets 79.1).

## Repository layout
```
src/absa/
  data.py      raw readers, sentence cropping, spaCy dependency parsing, splits
  graph.py     adjacency construction (dependency / sequential / random tree / fully connected), tree distances, position weights
  models.py    BiLSTM+Att, GCN (ASGCN-style), GAT, R-GAT, RoBERTa, RoBERTa+GCN, RoBERTa+R-GAT
  train.py     training / evaluation CLI (early stopping on a held-out dev split, saves metrics + predictions)
scripts/
  setup_resources.sh          download parsers, GloVe, datasets, RoBERTa (GitHub only)
  export_roberta_from_spacy.py convert RoBERTa-base inside en_core_web_trf to a HuggingFace checkpoint
  prepare_gold.py             build Rest14 / Lap14 / Yelp / MAMS (train/dev/test jsonl with parses)
  build_amazon2023_absa.py    distant-supervision ABSA set from Amazon Reviews 2023 + annotation pool
  make_amazon_gold.py         merge manual labels -> Amazon gold test set, audit of distant-label noise
  run_grid.py / run_all.sh    the full experiment schedule (resumable)
  analyze.py                  tables, significance tests, fine-grained analysis, figures
  mine_amazon_aspects.py      corpus-scale aspect mining over 100k Amazon reviews
  make_notebook.py            regenerates the notebook
data/processed/   ready-to-use jsonl datasets (tokens, heads, dependency labels, POS, aspect span, label)
data/annotations/ manual labels (Amazon gold set, distant-label audit)
results/          runs/, tables/, figures/, amazon_mining/
```

## How to run
Tested with Python 3.13, PyTorch 2.x (CPU), transformers 5.x, spaCy 3.8.

```bash
pip install -r requirements.txt
```
```bash
bash scripts/setup_resources.sh          # ~1.5 GB into ./resources
```
```bash
python scripts/prepare_gold.py --datasets rest14 lap14 yelp mams
```
```bash
python scripts/build_amazon2023_absa.py && python scripts/make_amazon_gold.py
```
```bash
bash scripts/run_all.sh                  # all experiments, ~12 h on 4 CPU cores; resumable
```
```bash
python scripts/analyze.py                # tables + figures in results/
```

`data/processed/` is committed, so the first three steps can be skipped if you only want to train.
Single runs:
```bash
cd src && python -m absa.train --dataset rest14 --model gcn --seed 1
```
```bash
cd src && python -m absa.train --dataset lap14 --model roberta-gcn --eval-on amazon23_gold
```
```bash
cd src && python -m absa.train --dataset lap14 --model gat --graph random   # ablation: random tree instead of syntax
```
With HuggingFace access you can use the original checkpoint via `--plm roberta-base`.
A GPU is not needed; with one, the RoBERTa runs take minutes instead of ~15–30 min each.

## Data
| dataset | source | labels | train / dev / test |
|---|---|---|---|
| **Yelp** | Yelp ABSA set from the PyABSA collection (reviews cropped to the aspect's sentence) | human | 727 / 81 / 245 |
| **Amazon23 (distant)** | Amazon Reviews 2023, *Electronics*, 40k reviews | distant supervision | 6805 / 837 / 858 |
| **Amazon23 (gold)** | 225 aspect mentions from 3k *disjoint* Amazon 2023 reviews; 154 from contrastive multi-aspect sentences | manual | – / – / 225 |
| Rest14, Lap14 | SemEval-2014 Task 4 | human | 3246/362/1120, 2095/233/638 |
| MAMS | Jiang et al. 2019: every sentence has ≥2 aspects with different polarities | human | 11186 / 1332 / 1336 |

**Amazon Reviews 2023 → ABSA.** The raw corpus has only star ratings. We mine an aspect lexicon
(frequent nouns + compound nouns such as *battery life*, *sound quality*), and label a sentence's aspects
positive if rating ≥ 4 ∧ VADER ≥ 0.5, negative if rating ≤ 2 ∧ VADER ≤ −0.3, neutral if the sentence contains
no sentiment word; contrastive sentences are excluded. A manual audit of 100 random distant test labels
(`results/tables/distant_label_audit.json`) finds 11 % invalid aspect candidates and **70 % label accuracy**
on valid ones; nearly all errors are sentence sentiment attributed to an incidental noun (→ should be
neutral). The gold set was labelled following SemEval-2014 guidelines (`data/annotations/`).

**Note on sources.** The compute environment used for this project could not reach HuggingFace or the
official dataset hosts, so all inputs are fetched from GitHub mirrors (pinned commits in
`scripts/setup_resources.sh`): the 100k-review Electronics subset of Amazon Reviews 2023, the PyABSA dataset
collection, GloVe (gensim-data) and spaCy models. The RoBERTa-base weights are extracted from spaCy's
`en_core_web_trf` pipeline (they were additionally fine-tuned by spaCy on OntoNotes tagging/parsing/NER).

## Models
All models get the same input (tokens, POS tags, spaCy dependency parse) and the same training protocol:
early stopping / model selection on a **held-out dev split** (never on test), 3 seeds for GloVe models.

| model | encoder | graph layer | readout |
|---|---|---|---|
| BiLSTM+Att | GloVe-300 (frozen) + BiLSTM | none | aspect-query attention over BiLSTM states |
| GCN | same | 2× GCN over the dependency graph, position-weighted (ASGCN, Zhang et al. 2019) | same |
| GAT | same | 2× 4-head graph attention restricted to dependency neighbours | same |
| R-GAT | same | GAT + dependency-relation embeddings in logits and messages | same |
| RoBERTa | RoBERTa-base, fine-tuned | none | `<s>` of `<s> sentence </s></s> aspect </s>` |
| RoBERTa+GCN / +R-GAT | RoBERTa-base, fine-tuned | 2× GCN / R-GAT over word states (first sub-word) | [`<s>`; mean of aspect node states] |
