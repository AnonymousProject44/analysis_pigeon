import os
import argparse
import itertools
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dynamic_leadership import load_directions, lead_delay, MAX_LAG_S, MIN_OVERLAP_S, CSV_DIR

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "../leadership")
MIN_CORR = 0.2       # pairs flying too differently carry no leadership meaning
MIN_TAU = 2          # frames; below this the pair is effectively synchronous

def build_network(dirs, frames, max_lag, min_overlap):
    """Directed graph: edge i->j (i leads j) weighted by the delay in frames."""
    G = nx.DiGraph()
    G.add_nodes_from(dirs)
    pairs = 0
    for i, j in itertools.combinations(sorted(dirs), 2):
        if len(np.intersect1d(frames[i], frames[j])) < min_overlap: continue
        tau, c = lead_delay(dirs[i], frames[i], dirs[j], frames[j], max_lag)
        if tau is None or c < MIN_CORR: continue
        pairs += 1
        if abs(tau) < MIN_TAU: continue          # synchronous: no directed edge
        a, b = (i, j) if tau > 0 else (j, i)
        G.add_edge(a, b, tau=abs(tau), corr=c)
    return G, pairs

MAX_TRIADS = 200000  # above this, sample triads at random (ttri is a proportion anyway)

def triangle_transitivity(G, rng=np.random.default_rng(0)):
    """Shizuka & McDonald (2012) ttri: 0 = random/cyclic, 1 = perfectly transitive.
    Triads are sampled on big networks; the estimate converges long before MAX_TRIADS."""
    nodes = list(G.nodes)
    n = len(nodes)
    total_triads = n * (n - 1) * (n - 2) // 6
    if total_triads <= MAX_TRIADS:
        triads = itertools.combinations(nodes, 3)
    else:
        triads = (rng.choice(n, 3, replace=False) for _ in range(MAX_TRIADS))
        triads = ([nodes[i] for i in t] for t in triads)

    tri, cyc = 0, 0
    for a, b, c in triads:
        if len({frozenset((x, y)) for x, y in itertools.permutations((a, b, c), 2)
                if G.has_edge(x, y)}) != 3:
            continue  # only triads where all three dyads are decided
        outdeg = sorted(sum(1 for y in (a, b, c) if G.has_edge(x, y)) for x in (a, b, c))
        if outdeg == [0, 1, 2]: tri += 1
        elif outdeg == [1, 1, 1]: cyc += 1
    total = tri + cyc
    if total == 0: return None, tri, cyc
    return (tri / total - 0.75) / 0.25, tri, cyc

def hierarchy_levels(G):
    """Layer the network: level 0 = top leaders. Cycles are collapsed (condensation)
    so a rock-paper-scissors trio shares a level instead of breaking the layering."""
    C = nx.condensation(G)
    order = list(nx.topological_sort(C))
    level_of_scc = {}
    for n in order:
        preds = list(C.predecessors(n))
        level_of_scc[n] = 0 if not preds else max(level_of_scc[p] for p in preds) + 1
    levels = {}
    for scc, lvl in level_of_scc.items():
        for bird in C.nodes[scc]['members']:
            levels[bird] = lvl
    return levels

def analyze(path, dt_s=0.005):
    tag = os.path.basename(path).replace('matching_', '').replace('_ts.csv', '')
    dirs, frames, pos = load_directions(path, dt_s)
    if len(dirs) < 3:
        return tag, None
    G, pairs = build_network(dirs, frames, int(MAX_LAG_S / dt_s), int(MIN_OVERLAP_S / dt_s))
    if G.number_of_edges() == 0:
        return tag, None

    ttri, n_tri, n_cyc = triangle_transitivity(G)
    levels = hierarchy_levels(G)

    rows = []
    for b in G.nodes:
        leads = list(G.successors(b))
        follows = list(G.predecessors(b))
        taus = [G[b][j]['tau'] for j in leads] + [-G[i][b]['tau'] for i in follows]
        rows.append({'bird': b, 'level': levels.get(b), 'n_leads': len(leads),
                     'n_follows': len(follows), 'mean_tau': float(np.mean(taus)) if taus else 0.0,
                     'mixed': len(leads) > 0 and len(follows) > 0})
    birds = pd.DataFrame(rows).sort_values(['level', 'n_leads'], ascending=[True, False])
    return tag, {'G': G, 'levels': levels, 'birds': birds, 'ttri': ttri,
                 'n_transitive': n_tri, 'n_cyclic': n_cyc,
                 'pairs_scored': pairs, 'top': birds.iloc[0]['bird'] if len(birds) else None}

def plot(tag, res):
    G, levels, birds = res['G'], res['levels'], res['birds']
    by_level = {}
    for b, l in levels.items():
        by_level.setdefault(l, []).append(b)
    # order each level by mean delay so the layout reads left-to-right as well
    rank = dict(zip(birds['bird'], birds['mean_tau']))
    pos = {}
    width = max(len(bs) for bs in by_level.values())
    for l, bs in by_level.items():
        bs = sorted(bs, key=lambda b: -rank.get(b, 0))
        for k, b in enumerate(bs):
            x = (k - (len(bs) - 1) / 2) * (width / max(len(bs), 1)) * 1.4
            pos[b] = (x, -l * 2.2)

    fig, ax = plt.subplots(figsize=(13, 1.6 * len(by_level) + 3))
    taus = [G[u][v]['tau'] for u, v in G.edges]
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=taus, edge_cmap=plt.cm.viridis,
                           arrowsize=7, width=0.7, alpha=0.35, node_size=420,
                           connectionstyle='arc3,rad=0.12')
    mixed = birds[birds['mixed']]['bird'].tolist()
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=420,
                           node_color=['tab:orange' if b in mixed else 'tab:blue' for b in G.nodes])
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=6, font_color='white')
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis,
                               norm=plt.Normalize(min(taus), max(taus)))
    fig.colorbar(sm, ax=ax, label='lead delay [frames]', shrink=0.7)
    tt = "n/a" if res['ttri'] is None else f"{res['ttri']:.2f}"
    ax.set_title(f"{tag}  |  leadership network  |  top Bird {res['top']}  |  "
                 f"{len(by_level)} levels  |  ttri {tt}  |  {res['n_cyclic']} cyclic triads\n"
                 f"orange = leads some birds and follows others (hidden by a mean ranking)",
                 fontsize=10)
    ax.set_ylabel("hierarchy level (top = leaders)")
    ax.axis('off')
    fig.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUT_DIR, f"network_{tag}.pdf"), bbox_inches='tight')
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('targets', nargs='*')
    parser.add_argument('--flights', type=str)
    parser.add_argument('--dt', type=int, default=5000)
    parser.add_argument('--force', action='store_true', help="Redo flights that already have a network PDF")
    args = parser.parse_args()

    tags = list(args.targets)
    if args.flights:
        tags += [l.strip() for l in open(args.flights) if l.strip()]

    dt_s = args.dt * 1e-6
    os.makedirs(OUT_DIR, exist_ok=True)
    summary, all_birds = [], []
    for t in tags:
        path = t if os.path.exists(t) else os.path.join(CSV_DIR, f"matching_{t}_ts.csv")
        if not os.path.exists(path):
            print(f"[WARN] not found: {t}"); continue
        if not args.force and os.path.exists(os.path.join(OUT_DIR, f"network_{t}.pdf")):
            print(f"{t}: skipped (network exists)"); continue
        tag, res = analyze(path, dt_s)
        if res is None:
            print(f"{tag}: no leadership edges"); continue
        plot(tag, res)
        b = res['birds']
        n_mixed = int(b['mixed'].sum())
        tt = "n/a" if res['ttri'] is None else f"{res['ttri']:.2f}"
        print(f"{tag}: top Bird {res['top']} | {b['level'].max()+1} levels | ttri {tt} | "
              f"{res['n_cyclic']} cyclic triads | {n_mixed}/{len(b)} birds lead-and-follow")
        summary.append({'flight': tag, 'top_bird': res['top'], 'birds': len(b),
                        'levels': int(b['level'].max()) + 1, 'edges': res['G'].number_of_edges(),
                        'ttri': res['ttri'], 'transitive_triads': res['n_transitive'],
                        'cyclic_triads': res['n_cyclic'], 'mixed_birds': n_mixed})
        b.insert(0, 'flight', tag)
        all_birds.append(b)

    if summary:
        pd.DataFrame(summary).to_csv(os.path.join(OUT_DIR, "network_summary.csv"), index=False)
        pd.concat(all_birds).to_csv(os.path.join(OUT_DIR, "network_birds.csv"), index=False)
        df = pd.DataFrame(summary)
        print(f"\n=== {len(df)} flights | median ttri {df['ttri'].median():.2f} | "
              f"{df['cyclic_triads'].sum()} cyclic triads total | "
              f"{df['mixed_birds'].sum()} lead-and-follow birds ===")
        print(f"Networks + CSVs in {OUT_DIR}")

if __name__ == "__main__":
    main()
