# Dependency-tree GNNs for aspect-based sentiment analysis

Course project: GCN and GAT over dependency parses vs. a fine-tuned RoBERTa baseline for aspect
sentiment classification, evaluated on Yelp and Amazon Reviews 2023 (plus SemEval-14 and MAMS for
comparison with prior work). Everything runs on CPU.

* Report: `report/report.pdf`
* Notebook with all results: `notebooks/ABSA_GNN_vs_Transformer.ipynb`
* Tables: `results/tables/`, figures: `results/figures/`, every run (metrics + per-example predictions): `results/runs/`

## Results

Test accuracy / macro-F1 (%). GloVe models: mean of 3 seeds. RoBERTa models: 2 seeds on Rest14, Lap14
and Yelp, 1 seed on MAMS and Amazon23. Model selection is always on a held-out dev split.

| model | Rest14 | Lap14 | MAMS | Yelp | Lap14 to Amazon23 gold |
|---|---|---|---|---|---|
| BiLSTM+Att (no graph) | 78.96 / 68.38 | 70.59 / 64.48 | 77.97 / 76.88 | 84.35 / 61.05 | 51.26 / 50.36 |
| GCN | 79.08 / 69.99 | 69.28 / 64.26 | 78.44 / 77.39 | 84.76 / 64.53 | 48.00 / 48.08 |
| GAT | 78.51 / 67.36 | 68.76 / 62.28 | 77.92 / 77.08 | 83.40 / 61.98 | 43.85 / 43.68 |
| R-GAT | 79.14 / 69.13 | 70.48 / 64.78 | 78.47 / 77.38 | 84.49 / 63.87 | 46.07 / 46.19 |
| RoBERTa | 84.02 / 74.69 | 77.66 / 72.76 | 83.46 / 83.08 | 81.02 / 51.92 | 59.11 / 57.32 |
| RoBERTa+GCN | 84.42 / 75.98 | 78.68 / 74.20 | n/a | 84.69 / 55.18 | 64.22 / 63.50 |
| RoBERTa+R-GAT | 84.11 / 75.57 | 78.13 / 73.94 | 83.23 / 82.83 | 83.67 / 57.64 | 63.11 / 61.86 |

Short version of the findings (details and caveats in the report):

* RoBERTa is 5 to 8 points better than every GloVe model on the benchmarks.
* On top of a GloVe BiLSTM the dependency graph does not help in-domain. A random tree with the same
  number of edges is clearly worse than the parse, but a plain left-to-right chain graph is as good.
* Adding a dependency GNN on top of RoBERTa helps on sentences whose aspects carry different polarities
  (Lap14: 44 to 60-62 %, Amazon gold: 47 to 59-62 %, consistent across both seeds). There is no gain on MAMS,
  where all 11k training sentences are of that kind.
* Labels derived from star ratings are about 70 % correct; models trained on them reach 93-96 % on a
  test set labelled the same way but only 49-54 % on the hand-labelled set.
* Over 97k Amazon reviews (588k aspect mentions) the GloVe R-GAT runs at about 2,200 mentions/s on 4 cores,
  RoBERTa+GCN at 77 mentions/s.

## Layout

```
src/absa/       data.py, graph.py, models.py, train.py
scripts/        data preparation, experiment grid, analysis, corpus mining, notebook generation
data/processed/ train/dev/test jsonl with parses (committed, so preprocessing can be skipped)
data/annotations/ labels for the Amazon gold set and the distant-label audit
results/        runs/, tables/, figures/, amazon_mining*/
report/         LaTeX source (ACL style) and PDF
```

## Running

```bash
pip install -r requirements.txt
```
```bash
bash scripts/setup_resources.sh
```
```bash
python scripts/prepare_gold.py --datasets rest14 lap14 yelp mams
```
```bash
python scripts/build_amazon2023_absa.py && python scripts/make_amazon_gold.py
```
```bash
bash scripts/run_all.sh
```
```bash
python scripts/analyze.py
```

`run_all.sh` runs the whole grid (about 12 h on 4 cores) and skips finished runs, so it can be restarted.
Single runs:

```bash
cd src && python -m absa.train --dataset rest14 --model gcn --seed 1
```
```bash
cd src && python -m absa.train --dataset lap14 --model roberta-gcn --eval-on amazon23_gold
```
```bash
cd src && python -m absa.train --dataset lap14 --model gat --graph random
```

Models: `lstm`, `gcn`, `gat`, `rgat`, `roberta`, `roberta-gcn`, `roberta-gat`.
Graphs: `dep`, `seq`, `random`, `full`.

## Data sources

The machine used for the experiments could not reach the HuggingFace Hub or the official dataset
hosts, so `setup_resources.sh` pulls everything from GitHub at pinned commits:

* Amazon Reviews 2023 (Hou et al., 2024): a 100k-review subset of the Electronics category mirrored on GitHub.
  With access to the official release, any category's reviews file can be converted to the same parquet columns.
* Yelp, SemEval-2014 and MAMS ABSA sets from the PyABSA dataset collection.
* GloVe 6B 300d (gensim-data release), spaCy `en_core_web_lg` for parsing.
* RoBERTa-base weights taken from spaCy's `en_core_web_trf` (`scripts/export_roberta_from_spacy.py`).
  These weights were further trained by spaCy on OntoNotes, so they differ slightly from the original
  checkpoint. With Hub access, pass `--plm roberta-base` instead.

The Amazon gold set (225 aspect mentions) and the 100-item audit of the distant labels are in
`data/annotations/`. See the report for how they were produced.
