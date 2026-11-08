"""Data loading, sentence cropping and dependency parsing for ABSA.

Every example is stored as one JSON object:
    {id, text, tokens, heads, deps, pos, from, to, aspect, label, sent_key}
where `heads[i]` is the index of token i's syntactic head (root points to itself),
`from`/`to` are the aspect's token span [from, to) and `label` is one of
LABELS. `sent_key` identifies the sentence (several aspects can share one).
"""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from typing import Iterable

LABELS = ["negative", "neutral", "positive"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}

_NLP = None


def get_nlp(model: str = "en_core_web_lg"):
    global _NLP
    if _NLP is None:
        import spacy
        _NLP = spacy.load(model, disable=["ner", "lemmatizer"])
    return _NLP


# ---------------------------------------------------------------- raw readers
def read_apc_txt(path: str):
    """PyABSA / ASGCN 3-line format: text-with-$T$ / aspect / polarity."""
    lines = [l.rstrip("\n") for l in open(path, encoding="utf-8")]
    out = []
    for i in range(0, len(lines) - 2, 3):
        text, aspect, pol = lines[i].strip(), lines[i + 1].strip(), lines[i + 2].strip()
        if "$T$" not in text or not aspect:
            continue
        pol = {"-1": "negative", "0": "neutral", "1": "positive"}.get(pol, pol).lower()
        if pol not in LABEL2ID:
            continue
        left, right = text.split("$T$", 1)
        full = left + aspect + right
        out.append({"text": full, "char_from": len(left), "char_to": len(left) + len(aspect),
                    "aspect": aspect, "label": pol})
    return out


# ---------------------------------------------------------------- parsing
def _doc_to_example(doc, cf, ct, aspect, label):
    span = doc.char_span(cf, ct, alignment_mode="expand")
    if span is None or len(span) == 0:
        return None
    toks = [t.text for t in doc]
    heads = [t.head.i - doc[0].i if t.head.i != t.i else t.i - doc[0].i for t in doc]
    return {
        "text": doc.text, "tokens": toks, "heads": heads,
        "deps": [t.dep_ for t in doc], "pos": [t.pos_ for t in doc],
        "from": span.start - doc[0].i, "to": span.end - doc[0].i,
        "aspect": aspect, "label": label,
    }


def crop_and_parse(raw: list[dict], max_tokens: int = 80, batch_size: int = 64,
                   n_process: int = 1, crop: bool = True):
    """Parse each example; if the text has several sentences, keep only the
    sentence(s) that contain the aspect (plus a neighbour if the sentence is
    very short). The cropped text is re-parsed so that the tree is a single
    rooted tree. Examples are de-duplicated on (text) for parsing speed."""
    nlp = get_nlp()
    texts = sorted({r["text"] for r in raw})
    docs = {t: d for t, d in zip(texts, nlp.pipe(texts, batch_size=batch_size, n_process=n_process))}

    cropped = []
    for r in raw:
        doc = docs[r["text"]]
        sents = list(doc.sents)
        if crop and len(sents) > 1:
            idx = [i for i, s in enumerate(sents)
                   if s.start_char < r["char_to"] and s.end_char > r["char_from"]]
            if not idx:
                continue
            lo, hi = min(idx), max(idx)
            while sum(len(sents[k]) for k in range(lo, hi + 1)) < 6 and (lo > 0 or hi < len(sents) - 1):
                if hi < len(sents) - 1:
                    hi += 1
                else:
                    lo -= 1
            s0 = sents[lo].start_char
            sub = doc.text[s0: sents[hi].end_char]
            cropped.append((sub, r["char_from"] - s0, r["char_to"] - s0, r))
        else:
            cropped.append((r["text"], r["char_from"], r["char_to"], r))

    texts2 = sorted({c[0] for c in cropped})
    docs2 = {t: d for t, d in zip(texts2, nlp.pipe(texts2, batch_size=batch_size, n_process=n_process))}
    out = []
    for sub, cf, ct, r in cropped:
        doc = docs2[sub]
        if len(doc) > max_tokens:  # keep a window around the aspect
            span = doc.char_span(cf, ct, alignment_mode="expand")
            if span is None:
                continue
            a = max(0, span.start - max_tokens // 2)
            b = min(len(doc), a + max_tokens)
            sub2 = doc[a:b].text
            cf2 = cf - doc[a].idx
            ct2 = ct - doc[a].idx
            doc = nlp(sub2)
            cf, ct = cf2, ct2
        ex = _doc_to_example(doc, cf, ct, r["aspect"], r["label"])
        if ex is None:
            continue
        for k in ("review_id", "rating", "parent_asin", "source"):
            if k in r:
                ex[k] = r[k]
        out.append(ex)
    return out


def add_sentence_keys(examples: list[dict], prefix: str):
    keymap = {}
    for i, ex in enumerate(examples):
        ex["id"] = f"{prefix}-{i}"
        ex["sent_key"] = keymap.setdefault(ex["text"], f"{prefix}-s{len(keymap)}")
    # multi-aspect statistics used in the analysis
    by_sent = defaultdict(list)
    for ex in examples:
        by_sent[ex["sent_key"]].append(ex["label"])
    for ex in examples:
        labs = by_sent[ex["sent_key"]]
        ex["n_aspects"] = len(labs)
        ex["mixed"] = len(set(labs)) > 1
    return examples


def stratified_split(examples, frac: float, seed: int = 13, group_key: str = "sent_key"):
    """Split off `frac` of the *sentences* (not aspects) so that aspects of the
    same sentence never leak across splits."""
    groups = defaultdict(list)
    for ex in examples:
        groups[ex[group_key]].append(ex)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    n_target = int(round(frac * len(examples)))
    a, b, n = [], [], 0
    for k in keys:
        if n < n_target:
            b.extend(groups[k]); n += len(groups[k])
        else:
            a.extend(groups[k])
    return a, b


# ---------------------------------------------------------------- io
def save_jsonl(examples: Iterable[dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def load_jsonl(path: str):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def label_dist(examples):
    c = Counter(ex["label"] for ex in examples)
    return {l: c.get(l, 0) for l in LABELS}
