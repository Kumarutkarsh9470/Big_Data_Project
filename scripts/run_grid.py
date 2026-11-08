"""Run a grid of experiments with N parallel workers; finished runs are skipped.

  python scripts/run_grid.py --grid main --workers 4
  python scripts/run_grid.py --grid ablation --workers 4
  python scripts/run_grid.py --grid plm --workers 1 --threads 4
"""
import argparse, itertools, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "results", "runs")
SEEDS = [1, 2, 3]


def run_name(ds, model, graph="dep", layers=2, tag="", seed=1):
    plain = model in ("lstm", "roberta")
    return f"{ds}__{model}" + (f"__{graph}" if not plain and graph != "dep" else "") + \
        (f"__L{layers}" if not plain and layers != 2 else "") + (f"__{tag}" if tag else "") + f"__s{seed}"


def job(ds, model, seed, graph="dep", layers=2, extra=(), eval_on=()):
    args = ["--dataset", ds, "--model", model, "--seed", str(seed), "--graph", graph, "--layers", str(layers)]
    if eval_on:
        args += ["--eval-on", *eval_on]
    return run_name(ds, model, graph, layers, "", seed), args + list(extra)


def grid(name):
    J = []
    if name == "main":
        for ds, m, s in itertools.product(["rest14", "yelp", "mams", "lap14", "amazon23"], ["lstm", "gcn", "gat", "rgat"], SEEDS):
            ev = {"lap14": ["amazon23", "amazon23_gold"], "amazon23": ["amazon23_gold", "lap14"],
                  "rest14": ["yelp"], "yelp": ["rest14"]}.get(ds, [])
            J.append(job(ds, m, s, eval_on=ev))
    elif name == "ablation":
        for ds, m, g, s in itertools.product(["rest14", "lap14", "mams"], ["gcn", "gat"], ["seq", "random", "full"], SEEDS):
            J.append(job(ds, m, s, graph=g))
        for ds, m, L, s in itertools.product(["rest14", "lap14"], ["gcn", "gat"], [1, 3, 4], SEEDS):
            J.append(job(ds, m, s, layers=L))
    elif name == "plm":
        for ds in ["yelp", "lap14", "rest14", "amazon23", "mams"]:
            ev = {"lap14": ["amazon23", "amazon23_gold"], "amazon23": ["amazon23_gold", "lap14"],
                  "rest14": ["yelp"], "yelp": ["rest14"]}.get(ds, [])
            for m in ["roberta", "roberta-gat", "roberta-gcn"]:
                J.append(job(ds, m, 1, eval_on=ev, extra=["--epochs", "3" if ds == "mams" else "4"]))
    elif name == "plm-seeds":
        for ds in ["yelp", "lap14", "rest14"]:
            for m in ["roberta", "roberta-gat", "roberta-gcn"]:
                J.append(job(ds, m, 2, extra=["--epochs", "4"]))
    else:
        raise ValueError(name)
    return J


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these datasets")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    logdir = os.path.join(ROOT, "results", "logs")
    os.makedirs(logdir, exist_ok=True)
    jobs = [(n, a) for n, a in grid(args.grid) if not os.path.exists(os.path.join(OUT, n + ".json"))]
    if args.only:
        jobs = [(n, a) for n, a in jobs if n.split("__")[0] in args.only]
    print(f"{len(jobs)} jobs to run", flush=True)

    def run(j):
        n, a = j
        t0 = time.time()
        with open(os.path.join(logdir, n + ".log"), "w") as log:
            r = subprocess.run([sys.executable, "-m", "absa.train", *a, "--threads", str(args.threads)],
                               cwd=os.path.join(ROOT, "src"), stdout=log, stderr=subprocess.STDOUT)
        print(f"{'ok ' if r.returncode == 0 else 'ERR'} {n} ({time.time()-t0:.0f}s)", flush=True)

    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(run, jobs))


if __name__ == "__main__":
    main()
