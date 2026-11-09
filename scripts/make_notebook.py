"""Generate notebooks/ABSA_GNN_vs_Transformer.ipynb (then execute it with nbconvert)."""
import os
import nbformat as nbf

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s))
code = lambda s: C.append(nbf.v4.new_code_cell(s))

md("""# Dependency-tree GNNs for Aspect-Based Sentiment Analysis
### GCN vs. GAT vs. a Transformer baseline on Amazon Reviews 2023 and Yelp

This notebook is the executable companion of the project. It
1. describes the datasets and the preprocessing (dependency parsing, aspect cropping, distant supervision for Amazon 2023),
2. shows how a dependency tree becomes the graph the GNNs operate on,
3. reports the main results, cross-domain transfer, ablations and fine-grained analyses (all numbers are read from `results/`),
4. runs a trained dependency-GAT on new sentences and shows what it attends to,
5. summarises corpus-scale aspect mining over 100k Amazon reviews.

**Re-running everything from scratch** (≈ 8–10 h on a 4-core CPU, no GPU needed) is described in `README.md`:
`bash scripts/setup_resources.sh && python scripts/prepare_gold.py --datasets rest14 lap14 yelp mams && python scripts/build_amazon2023_absa.py && python scripts/make_amazon_gold.py && bash scripts/run_all.sh`.
The cells below only need the already-produced `results/` (and, for the demo section, a saved checkpoint).""")

code("""import os, sys, json, glob
ROOT = os.path.abspath('..') if os.path.basename(os.getcwd()) == 'notebooks' else os.getcwd()
sys.path.insert(0, os.path.join(ROOT, 'src'))
import numpy as np, pandas as pd
from IPython.display import Markdown, Image, display
from absa.data import load_jsonl, label_dist
pd.set_option('display.width', 200)
def show_md(path): display(Markdown(open(os.path.join(ROOT, path)).read()))""")

md("## 1. Datasets")
md("""| name | source | labels | role |
|---|---|---|---|
| **Yelp** | Yelp reviews ABSA set (PyABSA collection) | human | target domain (restaurants/shops) |
| **Amazon23 (distant)** | Amazon Reviews 2023 – Electronics, 40k reviews | distant supervision (rating ∧ VADER, see below) | target domain (products), training |
| **Amazon23 (gold)** | 225 aspect mentions from *disjoint* Amazon 2023 reviews, 154 from contrastive multi-aspect sentences | manual | target domain evaluation |
| Rest14 / Lap14 | SemEval-2014 Task 4 | human | standard benchmarks (restaurant ≈ Yelp, laptop ≈ Amazon Electronics) |
| MAMS | Jiang et al. 2019 | human | stress test: every sentence has ≥2 aspects with different polarities |

Long Yelp reviews are cropped to the sentence(s) that contain the aspect, so that every example is a single dependency tree.""")
code("""rows = []
for ds in ['rest14', 'lap14', 'mams', 'yelp', 'amazon23']:
    for sp in ['train', 'dev', 'test']:
        d = load_jsonl(f'{ROOT}/data/processed/{ds}/{sp}.jsonl')
        rows.append({'dataset': ds, 'split': sp, 'n': len(d), **label_dist(d),
                     'avg tokens': round(np.mean([len(e['tokens']) for e in d]), 1),
                     '% multi-aspect sent.': round(100 * np.mean([e['n_aspects'] > 1 for e in d]), 1),
                     '% mixed-polarity sent.': round(100 * np.mean([e['mixed'] for e in d]), 1)})
g = load_jsonl(f'{ROOT}/data/processed/amazon23/gold_test.jsonl')
rows.append({'dataset': 'amazon23_gold', 'split': 'test', 'n': len(g), **label_dist(g),
             'avg tokens': round(np.mean([len(e['tokens']) for e in g]), 1),
             '% multi-aspect sent.': round(100 * np.mean([e['n_aspects'] > 1 for e in g]), 1),
             '% mixed-polarity sent.': round(100 * np.mean([e['mixed'] for e in g]), 1)})
pd.DataFrame(rows)""")

md("""### Distant supervision for Amazon Reviews 2023
Raw Amazon reviews only have star ratings. `scripts/build_amazon2023_absa.py` mines an aspect lexicon (frequent nouns and compound nouns such as *battery life*, *sound quality*) and labels a sentence's aspects
**positive** if rating ≥ 4 and VADER ≥ 0.5, **negative** if rating ≤ 2 and VADER ≤ −0.3, **neutral** if the sentence has no sentiment words at all; contrastive sentences (*but, however, …*) are excluded.
We hand-checked 100 random distant test labels to measure the noise:""")
code("""a = json.load(open(f'{ROOT}/results/tables/distant_label_audit.json'))
print(f"invalid aspect candidates: {a['invalid_rate']:.0%};  label accuracy on valid aspects: {a['label_accuracy_on_valid']:.1%}")
print('most common errors:', a['confusions'])""")
md("""Almost every error is *sentence sentiment attributed to an incidental noun* (neg/pos → neutral), polarity flips are rare.
The distant **test** set therefore has a noise ceiling of roughly 70 % accuracy, which is why the manually annotated **gold** set (with many contrastive, multi-aspect sentences) is the main Amazon evaluation.""")

md("## 2. From a sentence to a graph")
code("""import spacy, matplotlib.pyplot as plt
from absa.graph import adjacency, tree_distances
nlp = spacy.load('en_core_web_lg')
s = 'The battery life is great but the screen is too dim for outdoor use.'
doc = nlp(s)
for t in doc: print(f'{t.i:2d} {t.text:10s} --{t.dep_:>6s}--> {t.head.i:2d} {t.head.text}')
heads = [t.head.i for t in doc]
A = adjacency(heads, 'dep')
asp = [1, 2]   # 'battery life'
print('\\ntree distance from "battery life":', dict(zip([t.text for t in doc], tree_distances(heads, asp))))
fig, ax = plt.subplots(figsize=(5, 5)); ax.imshow(A, cmap='Blues')
ax.set_xticks(range(len(doc))); ax.set_xticklabels([t.text for t in doc], rotation=90)
ax.set_yticks(range(len(doc))); ax.set_yticklabels([t.text for t in doc]); ax.set_title('dependency adjacency (+ self loops)'); plt.show()""")
md("""In the linear order, *great* and *dim* are both exactly 3 tokens away from *screen*, so a purely sequential model has no positional cue for which opinion belongs to it. In the tree, *dim* is 2 hops from *screen* and *great* is 3 hops away (through the `conj` edge), while from *battery life* *great* is 2 hops away and *dim* 3. The GNNs propagate information only along these edges, so the opinion word that is *syntactically* attached to an aspect dominates its representation.

**Models** (`src/absa/models.py`):
* **BiLSTM+Att** – same encoder/readout as the GNNs, no graph (ablation baseline)
* **GCN** – ASGCN-style: BiLSTM → 2 GCN layers over the dependency graph (position-weighted) → aspect mask → attention readout
* **GAT** – the GCN layers replaced by 4-head graph attention restricted to dependency neighbours
* **R-GAT** – GAT whose attention logits and messages also use the dependency *relation* label (nsubj, amod, …)
* **RoBERTa** – fine-tuned sentence-pair transformer baseline `<s> sentence </s></s> aspect </s>`
* **RoBERTa+GCN / RoBERTa+R-GAT** – hybrid: RoBERTa word states → dependency GNN → [CLS; aspect] classifier""")

md("## 3. Main results")
code("show_md('results/tables/main_results.md')")
code("display(Image(f'{ROOT}/results/figures/main_acc.png')); display(Image(f'{ROOT}/results/figures/main_f1.png'))")
md("### Significance (paired bootstrap over test examples, correctness averaged over seeds)")
code("show_md('results/tables/significance.md')")
md("## 4. Cross-domain transfer")
code("show_md('results/tables/cross_domain.md')")
md("## 5. Ablations: does the *syntax* matter, or just having a graph?")
code("""show_md('results/tables/ablation.md')
if os.path.exists(f'{ROOT}/results/figures/ablation_graph.png'): display(Image(f'{ROOT}/results/figures/ablation_graph.png'))""")
md("## 6. Fine-grained analysis")
code("show_md('results/tables/fine_grained.md')")
code("""for f in sorted(glob.glob(f'{ROOT}/results/figures/distance_*.png')): display(Image(f))""")
md("## 7. Efficiency (CPU)")
code("show_md('results/tables/efficiency.md')")

md("## 8. Demo: trained models on new sentences\nFirst the GloVe R-GAT (trained on Lap14 + distant Amazon23) with its aspect-attention weights, then the RoBERTa+GCN hybrid (trained on Lap14).")
code("""import torch, argparse
from absa.models import GloveGraphModel
from absa.train import featurise, collate, DEPS
from absa.data import LABELS
ck = f'{ROOT}/results/models/lap14-amazon23_rgat.pt'
if os.path.exists(ck):
    ckpt = torch.load(ck, weights_only=False); a = argparse.Namespace(**ckpt['args'])
    model = GloveGraphModel(ckpt['emb'], len(DEPS), kind=a.model, layers=a.layers, heads=a.heads,
                            hidden=a.hidden, dropout=a.dropout, use_pos=not a.no_pos)
    model.load_state_dict(ckpt['state_dict']); model.eval()
    class V:
        stoi = {w: i for i, w in enumerate(ckpt['vocab'])}
        def ids(self, toks): return [self.stoi.get(t.lower(), 1) for t in toks]
    def analyse(sentence, aspect):
        doc = nlp(sentence); i = sentence.lower().index(aspect.lower())
        sp = doc.char_span(i, i + len(aspect), alignment_mode='expand')
        ex = {'tokens': [t.text for t in doc], 'heads': [t.head.i for t in doc], 'deps': [t.dep_ for t in doc],
              'pos': [t.pos_ for t in doc], 'from': sp.start, 'to': sp.end, 'aspect': aspect, 'label': None}
        f = featurise([ex], a, V(), None)
        with torch.no_grad(): p = torch.softmax(model(collate(f, torch.device('cpu'))), -1)[0].numpy()
        att = model.last_attention[0].numpy()
        top = sorted(zip(ex['tokens'], att), key=lambda x: -x[1])[:4]
        print(f'{aspect!r:18s} -> {LABELS[p.argmax()]:8s} {np.round(p, 2)}   attends to: ' + ', '.join(f'{w}({v:.2f})' for w, v in top))
    for s, asps in [('The battery life is great but the screen is too dim for outdoor use.', ['battery life', 'screen']),
                    ('The sound quality is amazing, although the ear cushions get uncomfortable after an hour.', ['sound quality', 'ear cushions']),
                    ('I bought this cable for my laptop and it stopped working after two days.', ['cable', 'laptop']),
                    ('Customer service never answered, but the keyboard itself works fine.', ['Customer service', 'keyboard'])]:
        print(s)
        for asp in asps: analyse(s, asp)
        print()
else:
    print('checkpoint not found; train it with: python -m absa.train --dataset lap14+amazon23 --model rgat --save-model ../results/models/lap14-amazon23_rgat.pt')""")

code("""ck2 = f'{ROOT}/results/models/lap14_roberta-gcn.pt'
if os.path.exists(ck2):
    from transformers import AutoTokenizer
    from absa.models import RobertaABSA
    c2 = torch.load(ck2, weights_only=False); a2 = argparse.Namespace(**c2['args'])
    tok = AutoTokenizer.from_pretrained(a2.plm)
    m2 = RobertaABSA(a2.plm, kind=a2.model, n_rel=len(DEPS), layers=a2.layers, heads=a2.heads)
    m2.load_state_dict(c2['state_dict']); m2.eval()
    def analyse2(sentence, aspect):
        doc = nlp(sentence); i = sentence.lower().index(aspect.lower())
        sp = doc.char_span(i, i + len(aspect), alignment_mode='expand')
        ex = {'tokens': [t.text for t in doc], 'heads': [t.head.i for t in doc], 'deps': [t.dep_ for t in doc],
              'pos': [t.pos_ for t in doc], 'from': sp.start, 'to': sp.end, 'aspect': aspect, 'label': None}
        with torch.no_grad(): p = torch.softmax(m2(collate(featurise([ex], a2, None, tok), torch.device('cpu'))), -1)[0].numpy()
        print(f'{aspect!r:18s} -> {LABELS[p.argmax()]:8s} {np.round(p, 2)}')
    for s, asps in [('The battery life is great but the screen is too dim for outdoor use.', ['battery life', 'screen']),
                    ('The sound quality is amazing, although the ear cushions get uncomfortable after an hour.', ['sound quality', 'ear cushions']),
                    ('I bought this cable for my laptop and it stopped working after two days.', ['cable', 'laptop']),
                    ('Customer service never answered, but the keyboard itself works fine.', ['Customer service', 'keyboard'])]:
        print(s)
        for asp in asps: analyse2(s, asp)
        print()""")

md("## 9. Corpus-scale aspect mining on Amazon Reviews 2023 (100k reviews)")
code("""p = f'{ROOT}/results/amazon_mining/summary.json'
if os.path.exists(p):
    print(json.dumps(json.load(open(p)), indent=1))
    display(Image(f'{ROOT}/results/amazon_mining/top_aspects.png')); display(Image(f'{ROOT}/results/amazon_mining/polarity_by_star.png'))
    display(pd.read_csv(f'{ROOT}/results/amazon_mining/aspect_summary.csv').head(25))""")
md("""### Accuracy vs. throughput at corpus scale
The same pipeline with the RoBERTa+GCN hybrid on the first 10k reviews: about 28x slower on CPU, but far more accurate on the gold set (≈64 % vs ≈49 %). Agreement between the two models on identical mentions is reported below.""")
code("""p = f'{ROOT}/results/amazon_mining_roberta_gcn_10k/summary.json'
if os.path.exists(p):
    r = json.load(open(p)); g = json.load(open(f'{ROOT}/results/amazon_mining/summary.json'))
    display(pd.DataFrame({'GloVe R-GAT (100k reviews)': [g['n_mentions'], round(g['timing']['mentions_per_s_inference']), round(g['timing']['reviews_per_s_end_to_end'], 1), g['spearman_review_aspect_sentiment_vs_stars']['rho']],
                          'RoBERTa+GCN (10k reviews)': [r['n_mentions'], round(r['timing']['mentions_per_s_inference']), round(r['timing']['reviews_per_s_end_to_end'], 1), r['spearman_review_aspect_sentiment_vs_stars']['rho']]},
                         index=['aspect mentions', 'mentions/s (inference)', 'reviews/s end-to-end', 'Spearman(aspect sentiment, stars)']))
    print('agreement of the two models on the same 10k reviews:', r.get('agreement_with_other_model'))""")

nb["cells"] = C
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
os.makedirs(os.path.join(ROOT, "notebooks"), exist_ok=True)
nbf.write(nb, os.path.join(ROOT, "notebooks", "ABSA_GNN_vs_Transformer.ipynb"))
print("written")
