"""Training / evaluation entry point.

Example:
  python -m absa.train --dataset rest14 --model gcn --seed 1
  python -m absa.train --dataset lap14 --model roberta --plm resources/roberta-base-spacy
  python -m absa.train --dataset amazon23 --model gat --graph random    # ablation
  python -m absa.train --dataset lap14 --model gcn --eval-on amazon23_gold  # cross-domain
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

from .data import LABEL2ID, LABELS, load_jsonl
from .graph import adjacency, position_weights
from .models import GloveGraphModel, RobertaABSA, count_params

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
GLOVE_MODELS = {"lstm", "gcn", "gat", "rgat"}
PLM_MODELS = {"roberta", "roberta-gcn", "roberta-gat"}

UPOS = ["<pad>", "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART",
        "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X", "SPACE"]
POS2ID = {p: i for i, p in enumerate(UPOS)}
# ClearNLP / spaCy English dependency labels (+ self loop, unknown)
DEPS = ["<pad>", "self", "<unk>", "ROOT", "acl", "acomp", "advcl", "advmod", "agent", "amod", "appos", "attr",
        "aux", "auxpass", "case", "cc", "ccomp", "compound", "conj", "csubj", "csubjpass", "dative", "dep",
        "det", "dobj", "expl", "intj", "mark", "meta", "neg", "nmod", "npadvmod", "nsubj", "nsubjpass",
        "nummod", "oprd", "parataxis", "pcomp", "pobj", "poss", "preconj", "predet", "prep", "prt", "punct",
        "quantmod", "relcl", "xcomp"]
DEP2ID = {d: i for i, d in enumerate(DEPS)}


def data_path(name, split):
    if name.endswith("_gold"):  # manually annotated test sets
        return os.path.join(ROOT, "data", "processed", name.replace("_gold", ""), f"gold_{split}.jsonl")
    return os.path.join(ROOT, "data", "processed", name, f"{split}.jsonl")


# ------------------------------------------------------------------ embeddings
def load_glove_subset(vocab_words, cache):
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return z["words"].tolist(), z["emb"]
    path = os.path.join(ROOT, "resources", "glove-wiki-gigaword-300.gz")
    want = set(vocab_words)
    found = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        first = f.readline().split()
        if len(first) > 2:  # no header line
            f.seek(0)
        for line in f:
            w, rest = line.split(" ", 1)
            if w in want:
                found[w] = np.asarray(rest.split(), dtype=np.float32)
    words = sorted(found)
    emb = np.stack([found[w] for w in words]) if words else np.zeros((0, 300), np.float32)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez(cache, words=np.array(words, dtype=object), emb=emb)
    return words, emb


class Vocab:
    def __init__(self, words, glove_words, glove_emb, seed=0):
        gi = {w: i for i, w in enumerate(glove_words)}
        self.itos = ["<pad>", "<unk>"] + sorted(set(words))
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        rng = np.random.RandomState(seed)
        d = glove_emb.shape[1] if len(glove_emb) else 300
        self.emb = rng.uniform(-0.25, 0.25, (len(self.itos), d)).astype(np.float32)
        self.emb[0] = 0
        self.coverage = 0
        for w, i in self.stoi.items():
            if w in gi:
                self.emb[i] = glove_emb[gi[w]]
                self.coverage += 1
        self.coverage /= max(len(self.itos) - 2, 1)

    def ids(self, toks):
        return [self.stoi.get(t.lower(), 1) for t in toks]


# ------------------------------------------------------------------ featurisation
def rel_matrix(ex, A):
    n = len(ex["tokens"])
    R = np.zeros((n, n), dtype=np.int64)
    R[A > 0] = DEP2ID["<unk>"]
    for i, h in enumerate(ex["heads"]):
        if h != i:
            r = DEP2ID.get(ex["deps"][i], DEP2ID["<unk>"])
            R[i, h] = r          # child -> head
            R[h, i] = r          # head -> child (same label; direction not modelled)
    np.fill_diagonal(R, DEP2ID["self"])
    return R


def featurise(examples, args, vocab=None, tok=None):
    feats = []
    for k, ex in enumerate(examples):
        n = len(ex["tokens"])
        A = adjacency(ex["heads"], args.graph, seed=k + 1000 * args.seed)
        f = {"n": n, "adj": A, "rel": rel_matrix(ex, A), "y": LABEL2ID[ex["label"]] if ex.get("label") else -1}
        amask = np.zeros(n, np.float32)
        amask[ex["from"]:ex["to"]] = 1
        f["amask"] = amask
        if vocab is not None:
            f["tok"] = vocab.ids(ex["tokens"])
            f["postag"] = [POS2ID.get(p, len(UPOS) - 1) for p in ex["pos"]]
            f["pw"] = position_weights(n, ex["from"], ex["to"])
        if tok is not None:
            ids, first = [tok.cls_token_id], []
            for i, w in enumerate(ex["tokens"]):
                piece = tok.encode((" " if i > 0 else "") + w, add_special_tokens=False) or [tok.unk_token_id]
                first.append(len(ids))
                ids.extend(piece)
            ids = ids[: args.max_subwords - 20]
            first = [min(x, len(ids) - 1) for x in first]
            asp = tok.encode(" " + ex["aspect"], add_special_tokens=False)[:16]
            ids = ids + [tok.sep_token_id, tok.sep_token_id] + asp + [tok.sep_token_id]
            f["ids"], f["first"] = ids, first
        feats.append(f)
    return feats


def collate(batch, device):
    B = len(batch)
    N = max(f["n"] for f in batch)
    out = {
        "adj": torch.zeros(B, N, N), "rel": torch.zeros(B, N, N, dtype=torch.long),
        "amask": torch.zeros(B, N), "mask": torch.zeros(B, N),
        "len": torch.tensor([f["n"] for f in batch]), "y": torch.tensor([f["y"] for f in batch]),
    }
    for i, f in enumerate(batch):
        n = f["n"]
        out["adj"][i, :n, :n] = torch.from_numpy(f["adj"])
        out["rel"][i, :n, :n] = torch.from_numpy(f["rel"])
        out["amask"][i, :n] = torch.from_numpy(f["amask"])
        out["mask"][i, :n] = 1
    if "tok" in batch[0]:
        out["tok"] = torch.zeros(B, N, dtype=torch.long)
        out["postag"] = torch.zeros(B, N, dtype=torch.long)
        out["pw"] = torch.zeros(B, N)
        for i, f in enumerate(batch):
            out["tok"][i, :f["n"]] = torch.tensor(f["tok"])
            out["postag"][i, :f["n"]] = torch.tensor(f["postag"])
            out["pw"][i, :f["n"]] = torch.from_numpy(f["pw"])
    if "ids" in batch[0]:
        L = max(len(f["ids"]) for f in batch)
        out["ids"] = torch.ones(B, L, dtype=torch.long)  # roberta pad id = 1
        out["att"] = torch.zeros(B, L, dtype=torch.long)
        out["first"] = torch.zeros(B, N, dtype=torch.long)
        for i, f in enumerate(batch):
            out["ids"][i, :len(f["ids"])] = torch.tensor(f["ids"])
            out["att"][i, :len(f["ids"])] = 1
            out["first"][i, :f["n"]] = torch.tensor(f["first"])
    return {k: v.to(device) for k, v in out.items()}


def batches(feats, bs, shuffle, rng):
    idx = list(range(len(feats)))
    if shuffle:
        # bucket by length for speed, then shuffle buckets
        rng.shuffle(idx)
        chunks = [sorted(idx[i:i + bs * 20], key=lambda j: feats[j]["n"]) for i in range(0, len(idx), bs * 20)]
        bl = [c[i:i + bs] for c in chunks for i in range(0, len(c), bs)]
        rng.shuffle(bl)
        return bl
    idx.sort(key=lambda j: feats[j]["n"])
    return [idx[i:i + bs] for i in range(0, len(idx), bs)]


@torch.no_grad()
def predict(model, feats, bs, device):
    model.eval()
    probs = [None] * len(feats)
    t0 = time.time()
    for b in batches(feats, bs, False, None):
        logits = model(collate([feats[j] for j in b], device))
        p = torch.softmax(logits, -1).cpu().numpy()
        for j, pj in zip(b, p):
            probs[j] = pj
    dt = time.time() - t0
    return np.stack(probs), dt


def metrics(y, p):
    return {"acc": accuracy_score(y, p), "f1": f1_score(y, p, average="macro", labels=[0, 1, 2], zero_division=0)}


# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", required=True, choices=sorted(GLOVE_MODELS | PLM_MODELS))
    ap.add_argument("--graph", default="dep", choices=["dep", "seq", "full", "random", "dep2"])
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=300)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--no-pos", action="store_true", help="disable POS-tag embeddings")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--bs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--plm", default=os.path.join(ROOT, "resources", "roberta-base-spacy"))
    ap.add_argument("--max-subwords", type=int, default=128)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--eval-on", nargs="*", default=[], help="extra test sets, e.g. amazon23_gold rest14")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "runs"))
    args = ap.parse_args(argv)

    torch.set_num_threads(args.threads)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device("cpu")
    is_plm = args.model in PLM_MODELS
    args.epochs = args.epochs or (5 if is_plm else 30)
    args.bs = args.bs or (16 if is_plm else 32)
    args.lr = args.lr or (2e-5 if is_plm else 1e-3)
    if args.model in ("lstm", "roberta"):
        args.graph = "none" if args.graph == "dep" else args.graph

    name = f"{args.dataset}__{args.model}" + (f"__{args.graph}" if args.model not in ("lstm", "roberta") and args.graph != "dep" else "") \
        + (f"__L{args.layers}" if args.model not in ("lstm", "roberta") and args.layers != 2 else "") + (f"__{args.tag}" if args.tag else "") \
        + f"__s{args.seed}"
    os.makedirs(args.out, exist_ok=True)
    out_json = os.path.join(args.out, name + ".json")

    train = load_jsonl(data_path(args.dataset, "train"))
    dev = load_jsonl(data_path(args.dataset, "dev"))
    tests = {args.dataset: load_jsonl(data_path(args.dataset, "test"))}
    for extra in args.eval_on:
        tests[extra] = load_jsonl(data_path(extra, "test"))
    g = args.graph if args.graph != "none" else "dep"
    args.graph = g

    vocab = tok = None
    if not is_plm:
        words = {t.lower() for ex in train + dev + [e for t in tests.values() for e in t] for t in ex["tokens"]}
        gw, ge = load_glove_subset(sorted(words), os.path.join(ROOT, "resources", "cache", f"glove_{args.dataset}_{'_'.join(sorted(args.eval_on))}.npz"))
        vocab = Vocab(words, gw, ge, seed=args.seed)
        model = GloveGraphModel(vocab.emb, len(DEPS), kind=args.model, layers=args.layers, heads=args.heads,
                                hidden=args.hidden, dropout=args.dropout, use_pos=not args.no_pos)
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.plm)
        model = RobertaABSA(args.plm, kind=args.model, n_rel=len(DEPS), layers=args.layers, heads=args.heads)

    ftr = featurise(train, args, vocab, tok)
    fdv = featurise(dev, args, vocab, tok)
    fte = {k: featurise(v, args, vocab, tok) for k, v in tests.items()}

    if is_plm:
        from transformers import get_linear_schedule_with_warmup
        enc_params = [p for n, p in model.named_parameters() if n.startswith("enc.")]
        head_params = [p for n, p in model.named_parameters() if not n.startswith("enc.")]
        opt = torch.optim.AdamW([{"params": enc_params, "lr": args.lr},
                                 {"params": head_params, "lr": 1e-3}], weight_decay=0.01)
        steps = args.epochs * ((len(ftr) + args.bs - 1) // args.bs)
        sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    else:
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-5)
        sched = None
    lossf = nn.CrossEntropyLoss()

    best, best_state, bad, history = -1, None, 0, []
    t_train = time.time()
    for ep in range(args.epochs):
        model.train()
        tot = 0.0
        for b in batches(ftr, args.bs, True, rng):
            bt = collate([ftr[j] for j in b], device)
            loss = lossf(model(bt), bt["y"])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if sched:
                sched.step()
            tot += loss.item() * len(b)
        pd, _ = predict(model, fdv, 64, device)
        m = metrics([f["y"] for f in fdv], pd.argmax(1))
        history.append({"epoch": ep + 1, "loss": tot / len(ftr), **{f"dev_{k}": v for k, v in m.items()}})
        print(f"[{name}] ep{ep+1} loss={tot/len(ftr):.4f} dev_acc={m['acc']:.4f} dev_f1={m['f1']:.4f}", flush=True)
        score = m["f1"] + m["acc"]
        if score > best:
            best, bad = score, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    train_time = time.time() - t_train
    model.load_state_dict(best_state)

    res = {"name": name, "args": vars(args), "history": history, "train_time_s": train_time,
           "n_params_trainable": count_params(model), "n_params_total": count_params(model, False),
           "glove_coverage": getattr(vocab, "coverage", None), "test": {}}
    preds = {}
    for k, ft in fte.items():
        p, dt = predict(model, ft, 64, device)
        y = [f["y"] for f in ft]
        res["test"][k] = {**metrics(y, p.argmax(1)), "n": len(ft), "infer_time_s": dt, "throughput": len(ft) / dt}
        preds[k] = [{"id": ex["id"], "gold": int(yy), "pred": int(pp.argmax()), "probs": [round(float(x), 4) for x in pp]}
                    for ex, yy, pp in zip(tests[k], y, p)]
    pdv, _ = predict(model, fdv, 64, device)
    res["dev"] = metrics([f["y"] for f in fdv], pdv.argmax(1))
    with open(out_json, "w") as f:
        json.dump(res, f, indent=1)
    with open(out_json.replace(".json", ".preds.json"), "w") as f:
        json.dump(preds, f)
    print(f"[{name}] DONE " + " ".join(f"{k}: acc={v['acc']:.4f} f1={v['f1']:.4f}" for k, v in res["test"].items())
          + f" | train {train_time:.0f}s", flush=True)
    return res


if __name__ == "__main__":
    main()
