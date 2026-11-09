"""Aggregate all runs into tables + figures (results/tables, results/figures).

  python scripts/analyze.py
"""
import glob, json, os, re, sys
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from absa.data import load_jsonl, LABEL2ID
from absa.graph import tree_distances

RUNS = os.path.join(ROOT, "results", "runs")
TAB = os.path.join(ROOT, "results", "tables")
FIG = os.path.join(ROOT, "results", "figures")
os.makedirs(TAB, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

MODEL_ORDER = ["lstm", "gcn", "gat", "rgat", "roberta", "roberta-gcn", "roberta-gat"]
MODEL_NAME = {"lstm": "BiLSTM+Att (no graph)", "gcn": "GCN (dep. tree)", "gat": "GAT (dep. tree)",
              "rgat": "R-GAT (dep. tree + rel.)", "roberta": "RoBERTa (fine-tuned)",
              "roberta-gcn": "RoBERTa+GCN", "roberta-gat": "RoBERTa+R-GAT"}
DS_ORDER = ["rest14", "lap14", "mams", "yelp", "amazon23", "amazon23_gold"]
DS_NAME = {"rest14": "Rest14", "lap14": "Lap14", "mams": "MAMS", "yelp": "Yelp",
           "amazon23": "Amazon23 (distant)", "amazon23_gold": "Amazon23 (gold)",
           "lap14+amazon23": "Lap14+Amazon23 (distant)"}


def parse_name(n):
    parts = n.split("__")
    d = {"train": parts[0], "model": parts[1], "graph": "dep", "layers": 2, "tag": "", "seed": int(parts[-1][1:])}
    for p in parts[2:-1]:
        if p in ("seq", "full", "random", "dep2"):
            d["graph"] = p
        elif re.fullmatch(r"L\d", p):
            d["layers"] = int(p[1:])
        else:
            d["tag"] = p
    return d


def load_runs():
    rows, preds = [], {}
    for f in glob.glob(os.path.join(RUNS, "*.json")):
        if f.endswith(".preds.json"):
            continue
        r = json.load(open(f))
        meta = parse_name(r["name"])
        for ds, m in r["test"].items():
            rows.append({**meta, "test": ds, "acc": m["acc"] * 100, "f1": m["f1"] * 100,
                         "train_time_s": r["train_time_s"], "params": r["n_params_trainable"],
                         "throughput": m["throughput"], "name": r["name"]})
        pf = f.replace(".json", ".preds.json")
        if os.path.exists(pf):
            preds[r["name"]] = json.load(open(pf))
    return pd.DataFrame(rows), preds


def fmt(m, s):
    return f"{m:.2f} ± {s:.2f}" if not np.isnan(s) else f"{m:.2f}"


def table(df, index, columns, metric):
    g = df.groupby([index, columns])[metric].agg(["mean", "std", "count"]).reset_index()
    g["cell"] = [fmt(a, b) for a, b in zip(g["mean"], g["std"].fillna(np.nan))]
    return g.pivot(index=index, columns=columns, values="cell"), g


# ------------------------------------------------------------------ main results
def main_results(df):
    d = df[(df.graph == "dep") & (df.layers == 2) & (df.tag == "") & (df.train == df.test)]
    out = []
    for metric in ["acc", "f1"]:
        t, g = table(d, "model", "test", metric)
        t = t.reindex([m for m in MODEL_ORDER if m in t.index])[[c for c in DS_ORDER if c in t.columns]]
        t.index = [MODEL_NAME[m] for m in t.index]
        t.columns = [DS_NAME[c] for c in t.columns]
        t.to_csv(os.path.join(TAB, f"main_{metric}.csv"))
        out.append((metric, t, g))
    n = d.groupby(["model", "test"]).seed.nunique().unstack()
    with open(os.path.join(TAB, "main_results.md"), "w") as f:
        for metric, t, _ in out:
            f.write(f"### Test {'accuracy' if metric == 'acc' else 'macro-F1'} (%), mean ± std over seeds\n\n")
            f.write(t.to_markdown() + "\n\n")
        f.write("Seeds per cell:\n\n" + n.to_markdown() + "\n")
    return out


def cross_domain(df):
    d = df[(df.graph == "dep") & (df.layers == 2) & (df.tag == "") & (df.train != df.test)]
    if d.empty:
        return None
    d = d.assign(pair=[f"{DS_NAME[a]} → {DS_NAME[b]}" for a, b in zip(d.train, d.test)])
    rows = {}
    for metric in ["acc", "f1"]:
        t, _ = table(d, "model", "pair", metric)
        t = t.reindex([m for m in MODEL_ORDER if m in t.index])
        t.index = [MODEL_NAME[m] for m in t.index]
        rows[metric] = t
    with open(os.path.join(TAB, "cross_domain.md"), "w") as f:
        for metric, t in rows.items():
            f.write(f"### Transfer {'accuracy' if metric == 'acc' else 'macro-F1'} (%)\n\n{t.to_markdown()}\n\n")
    return rows


def ablations(df):
    d = df[(df.train == df.test) & df.model.isin(["gcn", "gat"]) & (df.tag == "")]
    abl_ds = set(d[(d.graph != "dep") | (d.layers != 2)].test)
    d = d[d.test.isin(abl_ds)]
    df = df[df.test.isin(abl_ds)]
    with open(os.path.join(TAB, "ablation.md"), "w") as f:
        g = d[d.layers == 2].copy()
        base = df[(df.train == df.test) & (df.model == "lstm")].copy()
        base["graph"] = "none (BiLSTM)"
        g = pd.concat([g, base.assign(model="gcn"), base.assign(model="gat")])
        g["col"] = g.model + " / " + g.test
        for metric in ["acc", "f1"]:
            t, _ = table(g, "graph", "col", metric)
            order = [x for x in ["dep", "dep2", "seq", "random", "full", "none (BiLSTM)"] if x in t.index]
            t = t.reindex(order)
            f.write(f"### Graph structure ablation - {metric} (%)\n\n{t.to_markdown()}\n\n")
        L = d[d.graph == "dep"].copy()
        L["col"] = L.model + " / " + L.test
        for metric in ["acc", "f1"]:
            t, _ = table(L, "layers", "col", metric)
            f.write(f"### Number of graph layers - {metric} (%)\n\n{t.to_markdown()}\n\n")


def efficiency(df):
    d = df[(df.graph == "dep") & (df.layers == 2) & (df.tag == "") & (df.train == df.test)]
    e = d.groupby("model").agg(params=("params", "first"), train_time_s=("train_time_s", "mean"),
                               throughput=("throughput", "mean")).reindex([m for m in MODEL_ORDER if m in set(d.model)])
    e.index = [MODEL_NAME[m] for m in e.index]
    e["params"] = (e["params"] / 1e6).round(2)
    e = e.rename(columns={"params": "trainable params (M)", "train_time_s": "avg train time (s, CPU)",
                          "throughput": "inference (examples/s, CPU)"})
    e.round(1).to_markdown(open(os.path.join(TAB, "efficiency.md"), "w"))
    return e


# ------------------------------------------------------------------ fine-grained analysis
def opinion_lexicon():
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    return {w for w, v in SentimentIntensityAnalyzer().lexicon.items() if abs(v) >= 1.0 and w.isalpha()}


def example_features(ex, lex):
    n = len(ex["tokens"])
    asp = list(range(ex["from"], ex["to"]))
    op = [i for i, t in enumerate(ex["tokens"]) if t.lower() in lex and i not in asp]
    f = {"len": n, "n_aspects": ex.get("n_aspects", 1), "mixed": ex.get("mixed", False),
         "contrast": bool(re.search(r"\b(but|however|although|though|except|yet|whereas)\b", ex["text"], re.I))}
    if op:
        td = tree_distances(ex["heads"], asp)
        lin = [min(abs(i - a) for a in asp) for i in op]
        tre = [td[i] for i in op]
        f["lin_dist"] = min(lin)
        f["tree_dist"] = min(tre)
        # is the linearly-closest opinion word also (one of) the syntactically closest?
        j = int(np.argmin(lin))
        f["distractor"] = tre[j] > min(tre)
        f["lin_minus_tree"] = min(lin) - min(tre)
    else:
        f["lin_dist"] = f["tree_dist"] = np.nan
        f["distractor"] = False
        f["lin_minus_tree"] = np.nan
    return f


def per_example_correct(preds, df, test, models, train=None):
    """Average correctness over seeds for every example: {model: np.array}."""
    train = train or test
    out = {}
    for m in models:
        names = df[(df.model == m) & (df.train == train) & (df.test == test) & (df.graph == "dep") &
                   (df.layers == 2) & (df.tag == "")].name.unique()
        arrs = []
        for n in names:
            if n in preds and test in preds[n]:
                arrs.append(np.array([p["gold"] == p["pred"] for p in preds[n][test]], dtype=float))
        if arrs:
            out[m] = np.mean(arrs, 0)
    return out


def paired_bootstrap(a, b, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), (n, len(a)))
    diff = a[idx].mean(1) - b[idx].mean(1)
    return float((a.mean() - b.mean()) * 100), float((diff <= 0).mean())


def data_file(ds):
    return os.path.join(ROOT, "data", "processed", "amazon23", "gold_test.jsonl") if ds == "amazon23_gold" \
        else os.path.join(ROOT, "data", "processed", ds, "test.jsonl")


def fine_grained(df, preds):
    lex = opinion_lexicon()
    models = [m for m in MODEL_ORDER if m in set(df.model)]
    rows, sig = [], []
    all_feats = []
    for ds, train in [("rest14", "rest14"), ("lap14", "lap14"), ("mams", "mams"), ("yelp", "yelp"),
                      ("amazon23", "amazon23"), ("amazon23_gold", "amazon23"), ("amazon23_gold", "lap14")]:
        label = DS_NAME[ds] if train in (ds, "amazon23") else f"{DS_NAME[ds]} ← trained on {DS_NAME[train]}"
        if ds == "amazon23_gold" and train == "amazon23":
            label = f"{DS_NAME[ds]} ← trained on {DS_NAME[train]}"
        exs = load_jsonl(data_file(ds))
        feats = pd.DataFrame([example_features(e, lex) for e in exs])
        corr = per_example_correct(preds, df, ds, models, train=train)
        if not corr:
            continue
        for m, c in corr.items():
            feats[m] = c
        feats["dataset"] = ds if train in (ds, "amazon23") else f"{ds}_from_{train}"
        all_feats.append(feats)
        groups = {
            "all": np.ones(len(feats), bool),
            "single-aspect sentence": feats.n_aspects == 1,
            "multi-aspect sentence": feats.n_aspects > 1,
            "mixed-polarity sentence": feats.mixed,
            "contrast marker (but/however/...)": feats.contrast,
            "length <= 15": feats.len <= 15,
            "length 16-30": (feats.len > 15) & (feats.len <= 30),
            "length > 30": feats.len > 30,
            "opinion word: tree dist 1": feats.tree_dist == 1,
            "opinion word: tree dist 2": feats.tree_dist == 2,
            "opinion word: tree dist >= 3": feats.tree_dist >= 3,
            "linear dist >= 4": feats.lin_dist >= 4,
            "linear >> tree (lin - tree >= 3)": feats.lin_minus_tree >= 3,
            "distractor (lin-closest opinion not tree-closest)": feats.distractor,
        }
        for gname, mask in groups.items():
            mask = np.asarray(mask, bool)
            r = {"dataset": label, "group": gname, "n": int(mask.sum())}
            for m in corr:
                r[m] = corr[m][mask].mean() * 100 if mask.sum() else np.nan
            rows.append(r)
        # significance of key comparisons (paired bootstrap over test examples, seed-averaged)
        for a, b in [("gcn", "lstm"), ("gat", "lstm"), ("rgat", "lstm"), ("rgat", "gcn"), ("gat", "gcn"),
                     ("roberta", "rgat"), ("roberta-gat", "roberta"), ("roberta-gcn", "roberta")]:
            if a in corr and b in corr:
                d, p = paired_bootstrap(corr[a], corr[b])
                sig.append({"dataset": label, "A": MODEL_NAME[a], "B": MODEL_NAME[b],
                            "acc(A)-acc(B)": round(d, 2), "p (A<=B)": round(p, 4)})
    fg = pd.DataFrame(rows)
    fg = fg.rename(columns={m: MODEL_NAME[m] for m in models})
    fg.round(1).to_csv(os.path.join(TAB, "fine_grained.csv"), index=False)
    with open(os.path.join(TAB, "fine_grained.md"), "w") as f:
        for ds, g in fg.groupby("dataset", sort=False):
            f.write(f"### {ds}: accuracy (%) by subgroup (seed-averaged)\n\n")
            f.write(g.drop(columns="dataset").round(1).to_markdown(index=False) + "\n\n")
    sg = pd.DataFrame(sig)
    sg.to_markdown(open(os.path.join(TAB, "significance.md"), "w"), index=False)
    return fg, sg, (pd.concat(all_feats) if all_feats else None)


# ------------------------------------------------------------------ figures
def figures(df, feats):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    COLORS = {"lstm": "#9aa0a6", "gcn": "#4e79a7", "gat": "#59a14f", "rgat": "#2f7d32",
              "roberta": "#e15759", "roberta-gcn": "#f28e2b", "roberta-gat": "#b07aa1"}
    d = df[(df.graph == "dep") & (df.layers == 2) & (df.tag == "") & (df.train == df.test)]
    for metric, lab in [("acc", "Accuracy (%)"), ("f1", "Macro-F1 (%)")]:
        g = d.groupby(["test", "model"])[metric].agg(["mean", "std"]).reset_index()
        dss = [x for x in DS_ORDER if x in set(g.test)]
        ms = [m for m in MODEL_ORDER if m in set(g.model)]
        fig, ax = plt.subplots(figsize=(11, 4.2))
        w = 0.8 / len(ms)
        for k, m in enumerate(ms):
            sub = g[g.model == m].set_index("test").reindex(dss)
            ax.bar(np.arange(len(dss)) + k * w - 0.4 + w / 2, sub["mean"], w, yerr=sub["std"].fillna(0),
                   label=MODEL_NAME[m], color=COLORS[m], capsize=2, edgecolor="white", linewidth=0.5)
        ax.set_xticks(np.arange(len(dss)))
        ax.set_xticklabels([DS_NAME[x] for x in dss])
        ax.set_ylabel(lab)
        lo = np.nanmin(g["mean"]) - 8
        ax.set_ylim(max(0, lo), min(100, np.nanmax(g["mean"]) + 6))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(ncol=4, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.18))
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, f"main_{metric}.png"), dpi=200)
        plt.close(fig)

    if feats is not None:
        ms = [m for m in MODEL_ORDER if m in feats.columns]
        for ds in feats.dataset.unique():
            f = feats[feats.dataset == ds].copy()
            f = f[~f.tree_dist.isna()]
            f["tb"] = f.tree_dist.clip(upper=4).astype(int)
            f["lb"] = pd.cut(f.lin_dist, [0, 1, 2, 4, 8, 100], labels=["1", "2", "3-4", "5-8", ">8"])
            fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
            for ax, col, xl in [(axes[0], "tb", "tree distance aspect→nearest opinion word (4 = ≥4)"),
                                (axes[1], "lb", "linear distance aspect→nearest opinion word (tokens)")]:
                gg = f.groupby(col, observed=True)[ms].mean() * 100
                cnt = f.groupby(col, observed=True).size()
                for m in ms:
                    ax.plot(range(len(gg)), gg[m], marker="o", label=MODEL_NAME[m], color=COLORS[m])
                ax.set_xticks(range(len(gg)))
                ax.set_xticklabels([f"{i}\n(n={cnt[i]})" for i in gg.index], fontsize=8)
                ax.set_xlabel(xl, fontsize=9)
                ax.grid(alpha=0.3)
                ax.spines[["top", "right"]].set_visible(False)
            axes[0].set_ylabel("accuracy (%)")
            axes[1].legend(fontsize=7, frameon=False)
            fig.suptitle(DS_NAME.get(ds, ds.replace("_from_", " ← trained on ")), fontsize=10)
            fig.tight_layout()
            fig.savefig(os.path.join(FIG, f"distance_{ds}.png"), dpi=200)
            plt.close(fig)

    # ablation figure
    a = df[(df.train == df.test) & df.model.isin(["gcn", "gat"]) & (df.layers == 2) & (df.tag == "")]
    if a.graph.nunique() > 1:
        dss = [x for x in ["rest14", "lap14", "mams"] if x in set(a[a.graph != "dep"].test)]
        graphs = [x for x in ["dep", "seq", "random", "full"] if x in set(a.graph)]
        fig, axes = plt.subplots(1, len(dss), figsize=(4 * len(dss), 3.4), squeeze=False)
        for ax, ds in zip(axes[0], dss):
            for k, m in enumerate(["gcn", "gat"]):
                g = a[(a.test == ds) & (a.model == m)].groupby("graph").acc.agg(["mean", "std"]).reindex(graphs)
                ax.bar(np.arange(len(graphs)) + (k - 0.5) * 0.38, g["mean"], 0.38, yerr=g["std"].fillna(0),
                       label=m.upper(), color=COLORS[m], capsize=2)
            base = df[(df.train == ds) & (df.test == ds) & (df.model == "lstm")].acc.mean()
            ax.axhline(base, ls="--", color="#555", lw=1, label="BiLSTM (no graph)")
            ax.set_xticks(range(len(graphs)))
            ax.set_xticklabels(["dependency", "sequential", "random tree", "fully conn."][:len(graphs)] if graphs == ["dep", "seq", "random", "full"] else graphs, fontsize=8)
            ax.set_title(DS_NAME[ds], fontsize=10)
            vals = a[a.test == ds].groupby(["model", "graph"]).acc.mean()
            ax.set_ylim(min(vals.min(), base) - 3, vals.max() + 2)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0][0].set_ylabel("accuracy (%)")
        axes[0][-1].legend(fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "ablation_graph.png"), dpi=200)
        plt.close(fig)


def main():
    df, preds = load_runs()
    if df.empty:
        print("no runs yet")
        return
    df.to_csv(os.path.join(TAB, "all_runs.csv"), index=False)
    for metric, t, _ in main_results(df):
        print(f"\n== {metric}\n", t.to_string())
    cd = cross_domain(df)
    if cd:
        print("\n== cross-domain acc\n", cd["acc"].to_string())
        print("\n== cross-domain f1\n", cd["f1"].to_string())
    ablations(df)
    print("\n== efficiency\n", efficiency(df).round(1).to_string())
    fg, sg, feats = fine_grained(df, preds)
    figures(df, feats)
    print("\n== significance\n", sg.to_string() if len(sg) else "-")


if __name__ == "__main__":
    main()
