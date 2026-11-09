"""Run a trained model over all Amazon reviews and aggregate sentiment per aspect.
GloVe vocab is extended at inference time (embeddings are frozen in training)."""
import argparse, gzip, json, os, sys, time
from collections import Counter

import numpy as np
import pandas as pd
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from absa.data import LABELS, get_nlp
from absa.models import GloveGraphModel
from absa.train import featurise, collate, batches, DEPS, POS2ID, UPOS
from build_amazon2023_absa import clean, find_aspects


def load_lexicon(path):
    uni, bi = set(), set()
    for l in open(path):
        w = l.split("\t")[0]
        (bi if " " in w else uni).add(w)
    return uni, bi


def extend_vocab(ckpt, words):
    itos = list(ckpt["vocab"])
    emb = ckpt["emb"]
    known = set(itos)
    new = sorted({w for w in words if w not in known})
    want = set(new)
    found = {}
    with gzip.open(os.path.join(ROOT, "resources", "glove-wiki-gigaword-300.gz"), "rt", encoding="utf-8") as f:
        first = f.readline().split()
        if len(first) > 2:
            f.seek(0)
        for line in f:
            w, rest = line.split(" ", 1)
            if w in want:
                found[w] = np.asarray(rest.split(), dtype=np.float32)
    add = [w for w in new if w in found]
    itos += add
    emb = np.concatenate([emb, np.stack([found[w] for w in add])]) if add else emb
    return itos, emb, len(add), len(new)


class V:
    def __init__(self, itos):
        self.itos = itos
        self.stoi = {w: i for i, w in enumerate(itos)}

    def ids(self, toks):
        return [self.stoi.get(t.lower(), 1) for t in toks]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=os.path.join(ROOT, "resources/raw/amazon_reviews_2023_100k.parquet"))
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--lexicon", default=os.path.join(ROOT, "data/processed/amazon23/aspect_lexicon.txt"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "amazon_mining"))
    ap.add_argument("--n-process", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--compare-with", default=None, help="mentions.parquet of another model run on the same reviews (agreement)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_num_threads(4)
    timing = {}

    df = pd.read_parquet(args.parquet)
    df["text"] = df["text"].map(clean)
    df = df[df.text.str.len() >= 10].reset_index(drop=True)
    if args.limit:
        df = df.iloc[:args.limit]
    print("reviews:", len(df))

    nlp = get_nlp()
    lex_u, lex_b = load_lexicon(args.lexicon)
    t0 = time.time()
    mentions = []
    n_sents = 0
    for ri, doc in enumerate(nlp.pipe(df.text.tolist(), batch_size=256, n_process=args.n_process)):
        for s in doc.sents:
            n_sents += 1
            if not (3 <= len(s) <= 80):
                continue
            sd = s.as_doc()
            for fr, to in find_aspects(sd, lex_u, lex_b):
                mentions.append({"review": ri, "tokens": [t.text for t in sd], "heads": [t.head.i for t in sd],
                                 "deps": [t.dep_ for t in sd], "pos": [t.pos_ for t in sd],
                                 "from": fr, "to": to, "aspect": sd[fr:to].text.lower(), "label": None})
    timing["parse_s"] = time.time() - t0
    print(f"parsed {len(df)} reviews / {n_sents} sentences -> {len(mentions)} aspect mentions in {timing['parse_s']:.0f}s")

    ckpt = torch.load(args.model_path, weights_only=False)
    a = argparse.Namespace(**ckpt["args"])
    if a.model.startswith("roberta"):  # transformer / hybrid checkpoint
        from transformers import AutoTokenizer
        from absa.models import RobertaABSA
        tok = AutoTokenizer.from_pretrained(a.plm)
        model = RobertaABSA(a.plm, kind=a.model, n_rel=len(DEPS), layers=a.layers, heads=a.heads)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        vocab, bs = None, 32
    else:
        words = {t.lower() for m in mentions for t in m["tokens"]}
        itos, emb, n_add, n_new = extend_vocab(ckpt, words)
        print(f"vocab extended with {n_add}/{n_new} unseen words from GloVe")
        model = GloveGraphModel(emb, len(DEPS), kind=a.model, layers=a.layers, heads=a.heads,
                                hidden=a.hidden, dropout=a.dropout, use_pos=not a.no_pos)
        sd = ckpt["state_dict"]
        sd["emb.weight"] = torch.tensor(emb)
        model.load_state_dict(sd)
        model.eval()
        vocab, tok, bs = V(itos), None, 256

    t0 = time.time()
    feats = featurise(mentions, a, vocab, tok)
    timing["featurise_s"] = time.time() - t0
    t0 = time.time()
    probs = np.zeros((len(feats), 3), np.float32)
    with torch.no_grad():
        for b in batches(feats, bs, False, None):
            probs[b] = torch.softmax(model(collate([feats[j] for j in b], torch.device("cpu"))), -1).numpy()
    timing["inference_s"] = time.time() - t0
    timing["mentions_per_s_inference"] = len(feats) / timing["inference_s"]
    timing["reviews_per_s_end_to_end"] = len(df) / sum(timing[k] for k in ("parse_s", "featurise_s", "inference_s"))
    print(json.dumps(timing, indent=1))

    pred = probs.argmax(1)
    m = pd.DataFrame({"review": [x["review"] for x in mentions], "aspect": [x["aspect"] for x in mentions],
                      "pred": [LABELS[p] for p in pred], "conf": probs.max(1)})
    m["rating"] = df.rating.values[m.review.values]
    m["parent_asin"] = df.parent_asin.values[m.review.values]
    m["year"] = pd.to_datetime(df.timestamp.values[m.review.values]).year
    m.to_parquet(os.path.join(args.out, "mentions.parquet"))

    agg = m.groupby("aspect").agg(mentions=("pred", "size"),
                                  pos=("pred", lambda s: (s == "positive").mean()),
                                  neg=("pred", lambda s: (s == "negative").mean()),
                                  neu=("pred", lambda s: (s == "neutral").mean()),
                                  avg_rating=("rating", "mean")).sort_values("mentions", ascending=False)
    agg["net_sentiment"] = agg.pos - agg.neg
    agg.round(4).to_csv(os.path.join(args.out, "aspect_summary.csv"))

    val = m.groupby("pred").rating.agg(["mean", "count"]).reindex(LABELS)
    bucket = m.assign(star=m.rating).groupby("star").pred.value_counts(normalize=True).unstack().reindex(columns=LABELS)
    rv = m.assign(s=(m.pred == "positive").astype(int) - (m.pred == "negative").astype(int)).groupby("review").agg(s=("s", "mean"), rating=("rating", "first"))
    from scipy.stats import spearmanr
    rho, p = spearmanr(rv.s, rv.rating)
    summary = {"n_reviews": int(len(df)), "n_sentences": int(n_sents), "n_mentions": int(len(m)),
               "n_distinct_aspects": int(m.aspect.nunique()), "timing": timing,
               "pred_distribution": m.pred.value_counts(normalize=True).round(4).to_dict(),
               "avg_star_by_predicted_polarity": val["mean"].round(3).to_dict(),
               "spearman_review_aspect_sentiment_vs_stars": {"rho": round(float(rho), 4), "p": float(p), "n_reviews": int(len(rv))}}
    if args.compare_with:
        other = pd.read_parquet(args.compare_with)
        n = min(len(other), len(m))
        same = other.iloc[:n]
        assert (same.review.values == m.review.values[:n]).all()
        summary["agreement_with_other_model"] = round(float((same.pred.values == m.pred.values[:n]).mean()), 4)
    json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=1)
    bucket.round(4).to_csv(os.path.join(args.out, "polarity_by_star.csv"))
    print(json.dumps(summary, indent=1))
    print(bucket.round(3))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    top = agg[agg.mentions >= 200].head(30).sort_values("net_sentiment")
    fig, ax = plt.subplots(figsize=(7, 8))
    y = np.arange(len(top))
    ax.barh(y, -top.neg * 100, color="#e15759", label="negative")
    ax.barh(y, top.pos * 100, color="#4e79a7", label="positive")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{a} ({n})" for a, n in zip(top.index, top.mentions)], fontsize=8)
    ax.axvline(0, color="#333", lw=0.8)
    ax.set_xlabel("% of mentions predicted negative (left) / positive (right)")
    ax.set_title("Aspect sentiment, 30 most-mentioned aspects\n(Amazon Reviews 2023 Electronics, 100k reviews)", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "top_aspects.png"), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    bucket.plot(kind="bar", stacked=True, ax=ax, color=["#e15759", "#bab0ac", "#4e79a7"], width=0.75)
    ax.set_xlabel("review star rating")
    ax.set_ylabel("share of aspect mentions")
    ax.set_title("Predicted aspect polarity vs. review rating", fontsize=10)
    ax.legend(frameon=False, fontsize=8, bbox_to_anchor=(1, 1))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "polarity_by_star.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
