"""Merge manual annotations into the Amazon-2023 gold test set and measure the
noise of the distant labels.

  data/annotations/amazon23_gold_labels.tsv   -> data/processed/amazon23/gold_test.jsonl
  data/annotations/amazon23_distant_audit.tsv -> results/tables/distant_label_audit.json
"""
import json, os, sys
from collections import Counter
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from absa.data import load_jsonl, save_jsonl, add_sentence_keys, label_dist

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
M = {"pos": "positive", "neg": "negative", "neu": "neutral"}


def read_tsv(p):
    out = {}
    for l in open(p):
        if l.startswith("#") or not l.strip():
            continue
        k, v = l.split()
        out[k] = v
    return out


def main():
    pool = load_jsonl(f"{ROOT}/data/processed/amazon23/annotation_pool.jsonl")
    lab = read_tsv(f"{ROOT}/data/annotations/amazon23_gold_labels.tsv")
    gold = []
    for e in pool:
        l = lab.get(e["id"][-4:])
        if l is None or l == "x":
            continue
        e = dict(e)
        e["label"] = M[l]
        e["source"] = "manual"
        gold.append(e)
    orig_ids = [e["id"] for e in gold]
    gold = add_sentence_keys(gold, "amazon23_gold-test")
    for e, oid in zip(gold, orig_ids):
        e["pool_id"] = oid
    save_jsonl(gold, f"{ROOT}/data/processed/amazon23/gold_test.jsonl")
    print("gold test:", len(gold), label_dist(gold), "contrastive:", sum(e["contrast"] for e in gold),
          "mixed-polarity sentences:", sum(e["mixed"] for e in gold))

    test = {e["id"]: e for e in load_jsonl(f"{ROOT}/data/processed/amazon23/test.jsonl")}
    audit = read_tsv(f"{ROOT}/data/annotations/amazon23_distant_audit.tsv")
    n_x = sum(v == "x" for v in audit.values())
    pairs = [(test[k]["label"], M[v]) for k, v in audit.items() if v != "x"]
    agree = sum(a == b for a, b in pairs)
    per = {}
    for c in M.values():
        sel = [(a, b) for a, b in pairs if a == c]
        per[c] = {"n": len(sel), "precision": sum(a == b for a, b in sel) / max(len(sel), 1)}
    res = {"n_audited": len(audit), "invalid_aspect": n_x, "invalid_rate": n_x / len(audit),
           "label_accuracy_on_valid": agree / len(pairs), "per_distant_label": per,
           "confusions": Counter(f"{a}->{b}" for a, b in pairs if a != b).most_common()}
    os.makedirs(f"{ROOT}/results/tables", exist_ok=True)
    json.dump(res, open(f"{ROOT}/results/tables/distant_label_audit.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
