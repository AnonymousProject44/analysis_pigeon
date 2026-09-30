import os
import io
import argparse
import contextlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import correlate

from filter_trajectories3d import positions, CSV_DIR

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "../leadership")
MAX_LAG_S = 0.5          # search delays up to +/-0.5 s (pigeon reaction is 0.1-0.3 s)
MIN_OVERLAP_S = 0.75     # a pair needs this much common time to be scored
WINDOW_S = 1.5           # sliding-window length (tracks are short/fragmented)
STEP_S = 0.75            # window hop
WIN_COVER_S = 0.75       # a bird must cover at least this much of a window to count
MIN_BIRDS = 2            # a pair already has a well-defined follower/leader

def fit_plane(pts):
    """Best-fit 2D plane through the flock (SVD): centroid + two in-plane basis vectors."""
    c = pts.mean(axis=0)
    _, _, vh = np.linalg.svd(pts - c, full_matrices=False)
    return c, vh[0], vh[1]

def load_directions(path, dt_s=0.005):
    """Project trajectories onto the flock's best-fit 2D plane, then per-bird unit
    velocity directions and 2D positions on that plane (Nagy correlation runs in 2D)."""
    from stereo_visualizer import filter_outliers
    with contextlib.redirect_stdout(io.StringIO()):
        df = filter_outliers(pd.read_csv(path))
    all_pts = np.vstack([positions(g) for _, g in df.groupby('bird_id') if len(g) >= 20]) \
        if len(df) else np.zeros((1, 3))
    centroid, bu, bv = fit_plane(all_pts)

    dirs, frames, pos = {}, {}, {}
    for bid, g in df.groupby('bird_id'):
        g = g.sort_values('frame')
        if len(g) < 20: continue
        p3 = positions(g) - centroid
        p = np.column_stack([p3 @ bu, p3 @ bv])  # 2D plane coordinates
        v = np.gradient(p, axis=0)
        n = np.linalg.norm(v, axis=1, keepdims=True)
        n[n < 1e-9] = 1.0
        dirs[int(bid)] = v / n
        frames[int(bid)] = g.frame.to_numpy()
        pos[int(bid)] = p
    return dirs, frames, pos

def leader_edge_distance(leader, present, frames, pos, f0, f1):
    """Distance from the leader to the flock's convex-hull boundary in the fitted 2D
    plane, normalized by the flock radius (0 = on the edge). Mean position over [f0, f1)."""
    from scipy.spatial import ConvexHull, QhullError
    pts, ids = [], []
    for b in present:
        m = (frames[b] >= f0) & (frames[b] < f1)
        if m.sum() < 3: continue
        pts.append(pos[b][m].mean(axis=0))  # 2D plane coords
        ids.append(b)
    if leader not in ids or len(ids) < 3:
        return None
    pts = np.array(pts)
    try:
        hull = ConvexHull(pts)
    except QhullError:
        return None
    li = ids.index(leader)
    if li in hull.vertices:
        return 0.0
    verts = pts[hull.vertices]
    p = pts[li]
    d = min(np.linalg.norm(p - (a + np.clip(np.dot(p - a, b - a) / (np.dot(b - a, b - a) + 1e-9), 0, 1) * (b - a)))
            for a, b in zip(verts, np.roll(verts, -1, axis=0)))
    radius = np.linalg.norm(pts - pts.mean(axis=0), axis=1).mean() + 1e-9
    return float(d / radius)

def lead_delay(di, fi, dj, fj, max_lag):
    """Delay (frames, >0 => i leads j) maximizing direction correlation, or None."""
    common, ia, ib = np.intersect1d(fi, fj, return_indices=True)
    lag = min(max_lag, (len(common) - 5) // 2)  # cap the search to the available overlap
    if lag < 2: return None, 0.0
    a, b = di[ia], dj[ib]
    n = len(a)
    s = correlate(b[:, 0], a[:, 0], method='fft') + correlate(b[:, 1], a[:, 1], method='fft')
    s = s[n - 1 - lag:n + lag]
    c = s / (n - np.abs(np.arange(-lag, lag + 1)))
    k = int(np.argmax(c))
    return k - lag, float(c[k])

def leadership_scores(dirs, frames, max_lag, min_overlap):
    """Mean lead delay (frames) per bird over all sufficiently-overlapping pairs."""
    ids = sorted(dirs)
    score = {b: [] for b in ids}
    for i in ids:
        for j in ids:
            if j <= i: continue
            common = np.intersect1d(frames[i], frames[j])
            if len(common) < min_overlap: continue
            tau, c = lead_delay(dirs[i], frames[i], dirs[j], frames[j], max_lag)
            if tau is None or c < 0.2: continue  # ignore uncorrelated pairs
            score[i].append(tau)
            score[j].append(-tau)
    return {b: float(np.mean(v)) for b, v in score.items() if v}

def analyze(path, dt_s=0.005):
    tag = os.path.basename(path).replace('matching_', '').replace('_ts.csv', '')
    dirs, frames, pos = load_directions(path, dt_s)
    if len(dirs) < MIN_BIRDS:
        return tag, None

    max_lag = int(MAX_LAG_S / dt_s)
    min_ov = int(MIN_OVERLAP_S / dt_s)
    win = int(WINDOW_S / dt_s)
    step = int(STEP_S / dt_s)

    f0 = min(f[0] for f in frames.values())
    f1 = max(f[-1] for f in frames.values())

    global_score = leadership_scores(dirs, frames, max_lag, min_ov)
    if not global_score:
        return tag, None
    global_leader = max(global_score, key=global_score.get)

    cover = int(WIN_COVER_S / dt_s)
    windows = []
    for w0 in range(f0, f1 - win + 1, step):
        w1 = w0 + win
        wd, wf = {}, {}
        for b in dirs:
            m = (frames[b] >= w0) & (frames[b] < w1)
            if m.sum() >= cover:
                wd[b] = dirs[b][m]; wf[b] = frames[b][m]
        if len(wd) < MIN_BIRDS: continue
        sc = leadership_scores(wd, wf, max_lag, int(min_ov * 0.5))
        if not sc: continue
        windows.append({'t': (w0 + win / 2) * dt_s, 'leader': max(sc, key=sc.get), 'scores': sc,
                        'w0': w0, 'w1': w1, 'present': list(wd.keys())})

    leaders = [w['leader'] for w in windows]
    changes = sum(1 for a, b in zip(leaders, leaders[1:]) if a != b)
    n_distinct = len(set(leaders)) if leaders else 0
    dynamic = n_distinct > 1

    # convex-hull edge distance is only meaningful (and only computed) for dynamic flights
    edge_dists = None
    if dynamic:
        edge_dists = []
        for w in windows:
            d = leader_edge_distance(w['leader'], w['present'], frames, pos, w['w0'], w['w1'])
            w['edge_dist'] = d
            if d is not None: edge_dists.append(d)

    return tag, {'global_score': global_score, 'global_leader': global_leader,
                 'windows': windows, 'leaders': leaders, 'changes': changes,
                 'n_distinct': n_distinct, 'dynamic': dynamic, 'dt_s': dt_s,
                 'initial_leader': leaders[0] if leaders else None,
                 'final_leader': leaders[-1] if leaders else None,
                 'leader_edge_dist': (float(np.mean(edge_dists)) if edge_dists else None)}

def plot(tag, res):
    windows = res['windows']
    if not windows: return
    ids = sorted(res['global_score'])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), gridspec_kw={'height_ratios': [2, 1]})

    ts = [w['t'] for w in windows]
    for b in ids:
        ys = [w['scores'].get(b, np.nan) for w in windows]
        ax1.plot(ts, ys, marker='o', ms=3, lw=1.2, label=f"Bird {b}")
    ax1.axhline(0, color='0.7', lw=0.8, ls='--')
    ax1.set_ylabel("lead delay [frames]  (higher = leads)")
    edge_txt = "" if res['leader_edge_dist'] is None else f"  |  leader-to-hull-edge {res['leader_edge_dist']:.2f} R"
    ax1.set_title(f"{tag}  |  {'DYNAMIC' if res['dynamic'] else 'STATIC'} leadership  |  "
                  f"initial Bird {res['initial_leader']} -> final Bird {res['final_leader']}{edge_txt}")
    if len(ids) <= 12:
        ax1.legend(fontsize=7, ncol=2, loc='best')

    lead_ids = sorted(set(res['leaders']))
    ymap = {b: k for k, b in enumerate(lead_ids)}
    ax2.plot(ts, [ymap[w['leader']] for w in windows], drawstyle='steps-mid', marker='s', ms=5, zorder=1)
    # filled = leader on the hull edge, hollow (with distance) = interior; only for dynamic
    for w in windows:
        d = w.get('edge_dist')
        on_edge = d is not None and d < 0.05
        ax2.scatter(w['t'], ymap[w['leader']], s=70, zorder=2,
                    facecolor='tab:orange' if on_edge else 'white',
                    edgecolor='tab:orange' if d is not None else '0.6')
    ax2.set_yticks(range(len(lead_ids)))
    ax2.set_yticklabels([f"Bird {b}" for b in lead_ids])
    ax2.set_ylabel("current leader (filled = on hull edge)"); ax2.set_xlabel("time [s]")
    fig.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUT_DIR, f"leadership_{tag}.pdf"), bbox_inches='tight')
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('targets', nargs='*')
    parser.add_argument('--flights', type=str)
    parser.add_argument('--dt', type=int, default=5000, help="frame dt in microseconds")
    args = parser.parse_args()

    tags = list(args.targets)
    if args.flights:
        tags += [l.strip() for l in open(args.flights) if l.strip()]

    dt_s = args.dt * 1e-6
    os.makedirs(OUT_DIR, exist_ok=True)
    summary = []
    for t in tags:
        path = t if os.path.exists(t) else os.path.join(CSV_DIR, f"matching_{t}_ts.csv")
        if not os.path.exists(path):
            print(f"[WARN] not found: {t}"); continue
        tag, res = analyze(path, dt_s)
        if res is None:
            print(f"{tag}: not enough overlapping trajectories")
            summary.append({'flight': tag, 'verdict': 'undetermined', 'initial_leader': None,
                            'final_leader': None, 'changes': 0, 'leader_at_front_frac': None})
            continue
        plot(tag, res)
        verdict = "DYNAMIC" if res['dynamic'] else ("static" if res['n_distinct'] == 1 else "undetermined")
        if res['dynamic']:
            edge = "n/a" if res['leader_edge_dist'] is None else f"{res['leader_edge_dist']:.2f} R"
            print(f"{tag}: DYNAMIC | leader {res['initial_leader']} -> {res['final_leader']} | mean leader-to-hull-edge {edge}")
        else:
            print(f"{tag}: {verdict} | leader {res['initial_leader']}")
        summary.append({'flight': tag, 'verdict': verdict,
                        'initial_leader': res['initial_leader'], 'final_leader': res['final_leader'],
                        'changes': res['changes'], 'leader_edge_dist': res['leader_edge_dist']})

    if summary:
        df = pd.DataFrame(summary)
        df.to_csv(os.path.join(OUT_DIR, "leadership_summary.csv"), index=False)
        vc = df['verdict'].value_counts()
        print(f"\n=== {len(df)} flights | {vc.get('DYNAMIC', 0)} dynamic, "
              f"{vc.get('static', 0)} static, {vc.get('undetermined', 0)} undetermined ===")
        print(f"Plots + summary in {OUT_DIR}")

if __name__ == "__main__":
    main()
