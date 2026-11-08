#!/usr/bin/env bash
# Download everything the pipeline needs into ./resources (≈1.5 GB). Only GitHub is required.
set -e
cd "$(dirname "$0")/.."
mkdir -p resources/raw && cd resources
# spaCy parser (en_core_web_lg) + transformer pipeline that ships RoBERTa-base weights (en_core_web_trf)
for m in en_core_web_lg en_core_web_trf; do
  [ -f $m-3.8.0-py3-none-any.whl ] || curl -sSL -o $m-3.8.0-py3-none-any.whl \
    https://github.com/explosion/spacy-models/releases/download/$m-3.8.0/$m-3.8.0-py3-none-any.whl
  pip install -q $m-3.8.0-py3-none-any.whl
done
# GloVe 6B 300d (gensim-data mirror on GitHub)
[ -f glove-wiki-gigaword-300.gz ] || curl -sSL -o glove-wiki-gigaword-300.gz \
  https://github.com/RaRe-Technologies/gensim-data/releases/download/glove-wiki-gigaword-300/glove-wiki-gigaword-300.gz
# ABSA datasets (SemEval-14, MAMS, Yelp), pinned commit
if [ ! -d raw/ABSADatasets ]; then
  git clone -q https://github.com/yangheng95/ABSADatasets.git raw/ABSADatasets
  git -C raw/ABSADatasets checkout -q 676bcd3d88a901853c74acf364c9ad5c431b0e23
fi
# Amazon Reviews 2023 (Electronics) – 100k-review subset mirrored on GitHub, pinned commit.
# (Official source: https://amazon-reviews-2023.github.io ; to use it instead, convert any
#  category's reviews jsonl to parquet with columns parent_asin,rating,title,text,timestamp.)
[ -f raw/amazon_reviews_2023_100k.parquet ] || curl -sSL -o raw/amazon_reviews_2023_100k.parquet \
  https://raw.githubusercontent.com/kumaraguru123/SCC454-amazon-reviews-project/1b1d3576987fdccd4fe1b77a1416b13daf11cdd7/cleaned_data/reviews_clean.parquet
cd ..
# RoBERTa-base exported from en_core_web_trf to HuggingFace format (skip if you can use --plm roberta-base)
[ -d resources/roberta-base-spacy ] || python3 scripts/export_roberta_from_spacy.py --out resources/roberta-base-spacy
echo "resources ready"
