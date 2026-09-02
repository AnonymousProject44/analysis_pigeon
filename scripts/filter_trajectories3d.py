import os
import glob
import shutil
import argparse
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(SCRIPT_DIR, "../csv")
PLOT_DIR = os.path.join(SCRIPT_DIR, "../plots")

def positions(g):
    cols = ['x_m_smooth', 'y_m_smooth', 'z_m_smooth']
    if all(c in g for c in cols) and not g[cols[0]].isna().all():
        p = g[cols].copy()
        for c, raw in zip(cols, ['x_m', 'y_m', 'z_m']):
            p[c] = p[c].combine_first(g[raw])
        return p.to_numpy(dtype=float)
    return g[['x_m', 'y_m', 'z_m']].to_numpy(dtype=float)

def filter_matching(path, min_frames, min_disp_frac, max_angle_deg, from_orig=False):
    backup = path.replace('.csv', '.orig.csv')
    if not os.path.exists(backup):
        shutil.copy(path, backup)
    df = pd.read_csv(backup if from_orig else path)
    n0 = df.bird_id.nunique()

    span, disps = {}, {}
    for bid, g in df.groupby('bird_id'):
        g = g.sort_values('frame')
        span[bid] = (g.frame.iloc[0], g.frame.iloc[-1])
        p = positions(g)
        disps[bid] = p[-1] - p[0]

    long_ids = [b for b in span if span[b][1] - span[b][0] >= min_frames]
    if long_ids:
        med_disp = np.median([np.linalg.norm(disps[b]) for b in long_ids])
        passing = [b for b in long_ids if np.linalg.norm(disps[b]) >= min_disp_frac * med_disp]
    else:
        passing = []

    keep = set()
    for bid in passing:
        a0, a1 = span[bid]
        ref = np.zeros(3)
        for other in passing:
            if other == bid: continue
            b0, b1 = span[other]
            overlap = min(a1, b1) - max(a0, b0)
            if overlap > 0.3 * min(a1 - a0, b1 - b0):
                ref += disps[other]
        if np.linalg.norm(ref) < 1e-6:
            ref = np.sum([disps[b] for b in passing if b != bid], axis=0)
        u = disps[bid] / (np.linalg.norm(disps[bid]) + 1e-9)
        r = ref / (np.linalg.norm(ref) + 1e-9)
        ang = np.degrees(np.arccos(np.clip(np.dot(u, r), -1, 1)))
        if ang <= max_angle_deg:
            keep.add(bid)

    df[df.bird_id.isin(keep)].to_csv(path, index=False)

    sync_companions(path, keep, from_orig)
    return n0, len(keep), df[df.bird_id.isin(keep)]

def sync_companions(path, keep, from_orig=False):
    """Keep id_matches and bird_stats consistent with the surviving trajectory ids."""
    for prefix, col in (('id_matches_', 'Left_ID'), ('bird_stats_', 'bird_id')):
        comp = path.replace('matching_', prefix)
        if comp == path or not os.path.exists(comp): continue
        comp_bak = comp.replace('.csv', '.orig.csv')
        if not os.path.exists(comp_bak): shutil.copy(comp, comp_bak)
        c = pd.read_csv(comp_bak if from_orig else comp)
        c[c[col].isin(keep)].to_csv(comp, index=False)

def regen_plot(path, df_filtered):
    import matching_birds as mb
    from stereo_visualizer import filter_outliers
    tag = os.path.basename(path).replace('matching_', '').replace('_ts.csv', '')
    left_csv = os.path.join(CSV_DIR, f"tracking_ts_{tag}_left.csv")
    colors = {}
    if os.path.exists(left_csv):
        tl = pd.read_csv(left_csv)
        colors = {int(b): (g.iloc[0]['color_r'] / 255, g.iloc[0]['color_g'] / 255, g.iloc[0]['color_b'] / 255)
                  for b, g in tl.groupby('bird_id')}
    plot_path = os.path.join(PLOT_DIR, f"trajectories3d_{tag}_ts.pdf")
    dfp = filter_outliers(df_filtered)
    if not dfp.empty:
        mb.save_trajectory_plot_3d(dfp, colors, plot_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csvs', nargs='*', help="matching_*.csv files to filter")
    parser.add_argument('--flights', type=str, help="File with {date}_Spot{n}_clip_{xxx} tags")
    parser.add_argument('--min_frames', type=int, default=100, help="Minimum trajectory duration in frames")
    parser.add_argument('--min_disp_frac', type=float, default=0.2, help="Min 3D displacement as a fraction of the clip median")
    parser.add_argument('--max_angle', type=float, default=60.0, help="Max angle (deg) to the local flock direction")
    parser.add_argument('--no_plot', action='store_true', help="Skip regenerating the 3D plots")
    parser.add_argument('--from_orig', action='store_true', help="Reset from the .orig.csv backup before filtering")
    args = parser.parse_args()

    files = list(args.csvs)
    if args.flights:
        for tag in open(args.flights):
            if not tag.strip(): continue
            f = os.path.join(CSV_DIR, f"matching_{tag.strip()}_ts.csv")
            if os.path.exists(f):
                files.append(f)
            else:
                print(f"[WARN] no matching CSV for {tag.strip()}")

    if not files:
        print("Nothing to filter."); return
    total0, total1 = 0, 0
    for f in files:
        n0, n1, dff = filter_matching(f, args.min_frames, args.min_disp_frac, args.max_angle, args.from_orig)
        total0 += n0; total1 += n1
        print(f"{os.path.basename(f)}: {n0} -> {n1} trajectories")
        if not args.no_plot:
            regen_plot(f, dff)
    print(f"\n=== {len(files)} files | {total0} -> {total1} trajectories ({total0 - total1} removed) ===")

if __name__ == "__main__":
    main()
