"""Build the human-annotated ABSA datasets (train/dev/test jsonl with parses).

  rest14 : SemEval-2014 Task 4 Restaurants  (benchmark, Yelp-like domain)
  lap14  : SemEval-2014 Task 4 Laptops      (benchmark, product domain)
  yelp   : Yelp reviews ABSA set distributed with PyABSA (Yang et al.)
  mams   : MAMS (Jiang et al., 2019) -- every sentence has >=2 aspects with
           different polarities (optional, --with-mams)

Usage: python scripts/prepare_gold.py --absa-root resources/raw/ABSADatasets
"""
import argparse, os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from absa.data import (read_apc_txt, crop_and_parse, add_sentence_keys, stratified_split,
                       save_jsonl, label_dist)

D = "datasets/apc_datasets"
SOURCES = {
    "rest14": (f"{D}/110.SemEval/114.restaurant14/Restaurants_Train.xml.seg",
               f"{D}/110.SemEval/114.restaurant14/Restaurants_Test_Gold.xml.seg", None),
    "lap14": (f"{D}/110.SemEval/113.laptop14/Laptops_Train.xml.seg",
              f"{D}/110.SemEval/113.laptop14/Laptops_Test_Gold.xml.seg", None),
    "yelp": (f"{D}/119.Yelp/yelp.train.txt", f"{D}/119.Yelp/yelp.test.txt", None),
    "mams": (f"{D}/109.MAMS/train.xml.dat", f"{D}/109.MAMS/test.xml.dat", f"{D}/109.MAMS/valid.xml.dat"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--absa-root", default="resources/raw/ABSADatasets")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--datasets", nargs="+", default=["rest14", "lap14", "yelp"])
    ap.add_argument("--dev-frac", type=float, default=0.1)
    args = ap.parse_args()

    for name in args.datasets:
        t0 = time.time()
        tr, te, dv = SOURCES[name]
        train = crop_and_parse(read_apc_txt(os.path.join(args.absa_root, tr)))
        test = crop_and_parse(read_apc_txt(os.path.join(args.absa_root, te)))
        train = add_sentence_keys(train, f"{name}-train")
        test = add_sentence_keys(test, f"{name}-test")
        if dv:
            dev = add_sentence_keys(crop_and_parse(read_apc_txt(os.path.join(args.absa_root, dv))), f"{name}-dev")
        else:
            train, dev = stratified_split(train, args.dev_frac)
        od = os.path.join(args.out, name)
        os.makedirs(od, exist_ok=True)
        for split, exs in [("train", train), ("dev", dev), ("test", test)]:
            save_jsonl(exs, os.path.join(od, f"{split}.jsonl"))
            print(f"{name:7s} {split:5s} n={len(exs):5d} {label_dist(exs)}")
        print(f"  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
