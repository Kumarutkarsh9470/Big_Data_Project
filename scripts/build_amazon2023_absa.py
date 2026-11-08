"""Build an aspect-level sentiment dataset from raw Amazon Reviews 2023 (Electronics).

The raw corpus only has review-level star ratings, so aspect labels are derived
with *distant supervision* (documented in the report):

 1. Reviews are cleaned, split into sentences and dependency-parsed (spaCy).
 2. An aspect lexicon is mined from the corpus: frequent nouns and
    compound-noun bigrams (e.g. "battery life", "sound quality"), minus a
    stoplist of non-aspect nouns (time, thing, review, ...).
 3. For every sentence that mentions >=1 lexicon aspect:
       positive : review rating >= 4  AND VADER(sentence) >= +0.5
       negative : review rating <= 2  AND VADER(sentence) <= -0.3
       neutral  : sentence carries no sentiment words at all (VADER pos=neg=0)
                  and contains no evaluative adjective
    Sentences with contrast markers (but/however/although/...) are excluded
    from the distant set because their aspects may disagree in polarity.
 4. Reviews are split 80/10/10 into train/dev/test (by review, no leakage).
 5. From a *disjoint* pool of reviews we sample candidate aspect mentions for
    manual annotation (`annotation_pool.jsonl`), over-sampling multi-aspect and
    contrastive sentences, i.e. the cases where syntax should matter.

Usage: python scripts/build_amazon2023_absa.py --parquet resources/raw/amazon_reviews_2023_100k.parquet
"""
import argparse, html, os, random, re, sys, time
from collections import Counter, defaultdict

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from absa.data import save_jsonl, label_dist, add_sentence_keys, get_nlp

STOP_ASPECTS = set("""
thing things time times lot lots way ways bit day days year years month months week weeks
review reviews star stars amazon one ones problem problems issue issues reason reasons
fact kind sort type part parts couple bunch end point side lack need needs use uses
people person someone everything anything nothing something everyone anyone
hour hours minute minutes second seconds today yesterday tomorrow while
item items purchase purchases order orders product products buy money dollar dollars
job deal case cases price prices value
husband wife son daughter kid kids friend friends family mom dad
experience opinion idea question questions answer try work works difference
number amount rest top bottom front back inside outside
room home house place others hand hands night love book books course mind thanks half
none trouble pain plenty addition purpose choice solution chance past world results help
complaint complaints cons fun waste junk expectations market business school office bed
bedroom door floor kitchen living trip travel center area spot page note mine head eye
car gift christmas foot feet inch inches name tips user customer manufacturer company seller
""".split())
# 'product', 'item', 'price', 'value', 'case' are frequent but either generic or
# ambiguous ('case' = phone case vs. 'in this case'); price/value are kept below.
KEEP_ANYWAY = {"price", "value", "case"}
STOP_ASPECTS -= KEEP_ANYWAY

CONTRAST = re.compile(r"\b(but|however|although|though|except|yet|whereas|while|unfortunately|otherwise|only problem|only issue|only complaint|only downside)\b", re.I)


def clean(t: str) -> str:
    t = html.unescape(str(t))
    t = re.sub(r"<br\s*/?>", " ", t)
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def mine_lexicon(sent_docs, min_df=40, max_terms=400):
    uni, bi = Counter(), Counter()
    for d in sent_docs:
        seen_u, seen_b = set(), set()
        for t in d:
            if t.pos_ == "NOUN" and t.is_alpha and len(t.text) > 2:
                lw = t.text.lower()
                if lw not in STOP_ASPECTS:
                    seen_u.add(lw)
                if t.i > d[0].i:
                    p = d[t.i - d[0].i - 1]
                    if p.dep_ == "compound" and p.pos_ in ("NOUN", "PROPN") and p.is_alpha:
                        seen_b.add(f"{p.text.lower()} {lw}")
        uni.update(seen_u)
        bi.update(seen_b)
    lex_u = [w for w, c in uni.most_common() if c >= min_df][:max_terms]
    bad_bi = {"christmas gift", "living room", "home office", "name brand", "brand name", "price range",
              "price tag", "return window", "learning curve", "quality sound", "quality cable", "youtube videos",
              "youtube video", "water bottle", "inch laptop", "gb card", "c port", "c ports", "c cable", "ray player",
              "zip ties", "heart rate", "air bubbles", "water proof", "ear canal"}
    lex_b = [w for w, c in bi.most_common() if c >= min_df // 2 and w not in bad_bi
             and w.split()[1] not in STOP_ASPECTS - {"service", "customer"}][:max_terms // 2]
    lex_b.extend(["customer service", "customer support", "tech support"])
    return set(lex_u), set(lex_b), uni, bi


def find_aspects(d, lex_u, lex_b):
    """Return list of (from, to) token spans, preferring bigrams."""
    toks = [t.text.lower() for t in d]
    spans, used = [], set()
    for i in range(len(toks) - 1):
        if f"{toks[i]} {toks[i+1]}" in lex_b and d[i + 1].pos_ == "NOUN":
            spans.append((i, i + 2)); used.update([i, i + 1])
    for i, t in enumerate(d):
        if i in used:
            continue
        if toks[i] in lex_u and t.pos_ == "NOUN":
            spans.append((i, i + 1))
    return sorted(spans)


def sent_to_example(d, fr, to, label, meta):
    base = d[0].i
    heads = [t.head.i - base for t in d]
    return {"text": d.text, "tokens": [t.text for t in d], "heads": heads,
            "deps": [t.dep_ for t in d], "pos": [t.pos_ for t in d],
            "from": fr, "to": to, "aspect": d[fr:to].text, "label": label, **meta}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="resources/raw/amazon_reviews_2023_100k.parquet")
    ap.add_argument("--out", default="data/processed/amazon23")
    ap.add_argument("--n-reviews", type=int, default=40000)
    ap.add_argument("--pool-reviews", type=int, default=3000, help="disjoint reviews for manual annotation")
    ap.add_argument("--cap", type=int, nargs=3, default=[3000, 2000, 3500], help="max neg/neu/pos")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--n-hard-sents", type=int, default=60)
    ap.add_argument("--pool-size", type=int, default=260)
    ap.add_argument("--n-process", type=int, default=4)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    os.makedirs(args.out, exist_ok=True)

    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    vader = SentimentIntensityAnalyzer()

    df = pd.read_parquet(args.parquet)
    df["text"] = df["text"].map(clean)
    df = df[df.text.str.len().between(20, 1500)].drop_duplicates("text").reset_index(drop=True)
    df["review_id"] = ["amz-r%06d" % i for i in range(len(df))]
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    pool_df = df.iloc[:args.pool_reviews]
    main_df = df.iloc[args.pool_reviews: args.pool_reviews + args.n_reviews]
    print("reviews: main", len(main_df), "annotation pool", len(pool_df), "rating dist", main_df.rating.value_counts().to_dict())

    nlp = get_nlp()
    t0 = time.time()

    def parse(frame):
        out = []
        for doc, (rid, rating, asin) in zip(
                nlp.pipe(frame.text.tolist(), batch_size=128, n_process=args.n_process),
                zip(frame.review_id, frame.rating, frame.parent_asin)):
            for s in doc.sents:
                sd = s.as_doc()
                if 5 <= len(sd) <= 60:
                    out.append((sd, {"review_id": rid, "rating": int(rating), "parent_asin": asin}))
        return out

    main_sents = parse(main_df)
    pool_sents = parse(pool_df)
    print(f"parsed {len(main_sents)} + {len(pool_sents)} sentences in {time.time()-t0:.0f}s")

    lex_u, lex_b, uni, bi = mine_lexicon([s for s, _ in main_sents])
    with open(os.path.join(args.out, "aspect_lexicon.txt"), "w") as f:
        for w in sorted(lex_b, key=lambda w: -bi[w]):
            f.write(f"{w}\t{bi[w]}\n")
        for w in sorted(lex_u, key=lambda w: -uni[w]):
            f.write(f"{w}\t{uni[w]}\n")
    print("lexicon:", len(lex_u), "unigrams,", len(lex_b), "bigrams; top:", sorted(lex_u, key=lambda w: -uni[w])[:40])

    # ---------------- distant labelling
    stats = Counter()
    by_label = defaultdict(list)
    for d, meta in main_sents:
        spans = find_aspects(d, lex_u, lex_b)
        if not spans:
            continue
        stats["sent_with_aspect"] += 1
        if CONTRAST.search(d.text):
            stats["skip_contrast"] += 1
            continue
        v = vader.polarity_scores(d.text)
        r = meta["rating"]
        has_adj = any(t.pos_ == "ADJ" and t.dep_ in ("acomp", "amod", "advmod", "attr", "conj") and
                      t.text.lower() not in {"other", "same", "new", "first", "last", "many", "few", "more", "much", "own", "several"}
                      for t in d)
        if r >= 4 and v["compound"] >= 0.5:
            lab = "positive"
        elif r <= 2 and v["compound"] <= -0.3:
            lab = "negative"
        elif v["pos"] == 0 and v["neg"] == 0 and not has_adj and "!" not in d.text and "?" not in d.text:
            lab = "neutral"
        else:
            stats["skip_ambiguous"] += 1
            continue
        for fr, to in spans:
            by_label[lab].append(sent_to_example(d, fr, to, lab, meta))
    print("labelling stats:", dict(stats), {k: len(v) for k, v in by_label.items()})

    # cap per class, sampling whole reviews
    selected = []
    for lab, cap in zip(["negative", "neutral", "positive"], args.cap):
        exs = by_label[lab]
        rng.shuffle(exs)
        # at most 2 aspects per sentence to avoid long lists of the same sentence
        per_sent = Counter()
        keep = []
        for e in exs:
            if per_sent[e["text"]] >= 2:
                continue
            per_sent[e["text"]] += 1
            keep.append(e)
            if len(keep) >= cap:
                break
        selected.extend(keep)

    reviews = sorted({e["review_id"] for e in selected})
    rng.shuffle(reviews)
    n = len(reviews)
    split_of = {r: ("test" if i < 0.1 * n else "dev" if i < 0.2 * n else "train") for i, r in enumerate(reviews)}
    splits = defaultdict(list)
    for e in selected:
        e["source"] = "distant"
        splits[split_of[e["review_id"]]].append(e)
    for s in ("train", "dev", "test"):
        exs = add_sentence_keys(splits[s], f"amazon23-{s}")
        save_jsonl(exs, os.path.join(args.out, f"{s}.jsonl"))
        print(f"amazon23 {s:5s} n={len(exs):5d} {label_dist(exs)}")

    # ---------------- annotation pool (disjoint reviews)
    cands = []
    for d, meta in pool_sents:
        spans = find_aspects(d, lex_u, lex_b)
        if not spans or len(d) > 45:
            continue
        contrast = bool(CONTRAST.search(d.text))
        for fr, to in spans:
            e = sent_to_example(d, fr, to, None, meta)
            e["contrast"] = contrast
            e["n_spans"] = len(spans)
            cands.append(e)
    rng.shuffle(cands)
    hard = [c for c in cands if c["contrast"] and c["n_spans"] >= 2]
    easy = [c for c in cands if not (c["contrast"] and c["n_spans"] >= 2)]
    # keep whole sentences for the hard part so that contrasting aspects are both annotated
    hard_sents = []
    seen = set()
    for c in hard:
        if c["text"] not in seen:
            seen.add(c["text"]); hard_sents.append(c["text"])
    hard_keep = set(hard_sents[:args.n_hard_sents])
    pool = [c for c in hard if c["text"] in hard_keep]
    seen_easy = set()
    for c in easy:
        if len(pool) >= args.pool_size:
            break
        if c["text"] in seen_easy:
            continue
        seen_easy.add(c["text"])
        pool.append(c)
    for i, c in enumerate(pool):
        c["id"] = f"amz-gold-{i:04d}"
    save_jsonl(pool, os.path.join(args.out, "annotation_pool.jsonl"))
    print("annotation pool:", len(pool), "of which contrastive multi-aspect:", sum(c["contrast"] and c["n_spans"] >= 2 for c in pool))


if __name__ == "__main__":
    main()
