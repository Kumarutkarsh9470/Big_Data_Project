import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc

toks = ["The", "battery", "life", "is", "great", "but", "the", "screen", "is", "too", "dim", "for", "outdoor", "use"]
heads = [2, 2, 3, 3, 3, 3, 7, 8, 3, 10, 8, 10, 13, 11]
deps = ["det", "compound", "nsubj", "ROOT", "acomp", "cc", "det", "nsubj", "conj", "advmod", "acomp", "prep", "amod", "pobj"]
POS, NEG = "#2A9D8F", "#E76F51"
color = {1: POS, 2: POS, 4: POS, 7: NEG, 10: NEG}
path_pos = {(2, 3), (4, 3)}
path_neg = {(7, 8), (10, 8)}


def draw(fname, dark):
    fg = "#FFFFFF" if dark else "#1F2933"
    mute = "#8D99AE" if dark else "#9AA5B1"
    bg = "#14213D" if dark else "#FFFFFF"
    fig, ax = plt.subplots(figsize=(14, 3.6))
    xs, x = [], 0.0
    for t in toks:
        w = 0.16 * len(t) + 0.55
        xs.append(x + w / 2)
        x += w
    arcs = sorted([(min(i, h), max(i, h), i, h) for i, h in enumerate(heads) if h != i], key=lambda a: a[1] - a[0])
    level = {}
    for a, b, i, h in arcs:
        inner = [level[(c, d)] for (c, d) in level if a <= c and d <= b and (c, d) != (a, b)]
        level[(a, b)] = 1 + max(inner, default=0)
    for a, b, i, h in arcs:
        lv = level[(a, b)]
        hgt = 0.45 * lv
        if (i, h) in path_pos:
            ec, lw, tc = POS, 3.4, POS
        elif (i, h) in path_neg:
            ec, lw, tc = NEG, 3.4, NEG
        else:
            ec, lw, tc = mute, 1.3, mute
        w = xs[b] - xs[a]
        ax.add_patch(Arc(((xs[a] + xs[b]) / 2, 0.22), w, hgt * 2, theta1=0, theta2=180, color=ec, lw=lw))
        ax.text((xs[a] + xs[b]) / 2, 0.22 + hgt, deps[i], ha="center", va="center", fontsize=11, color=tc,
                family="DejaVu Sans", bbox=dict(boxstyle="round,pad=0.15", fc=bg, ec="none"))
    for i, t in enumerate(toks):
        c = color.get(i, fg)
        ax.text(xs[i], 0, t, ha="center", va="top", fontsize=18, color=c,
                fontweight="bold" if i in color else "normal", family="DejaVu Sans")
    ax.set_xlim(-0.2, x + 0.2)
    ax.set_ylim(-0.55, 0.22 + 0.45 * max(level.values()) + 0.3)
    ax.set_aspect("auto")
    ax.axis("off")
    fig.savefig(fname, dpi=200, bbox_inches="tight", transparent=not dark, facecolor=bg if dark else "none")
    plt.close(fig)


draw("assets/dep_dark.png", True)
draw("assets/dep_light.png", False)
print("ok")
