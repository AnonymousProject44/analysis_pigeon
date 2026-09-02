import os
import glob
import shutil
import argparse
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(SCRIPT_DIR, "../csv")

def filter_csv(path, min_frames, min_disp_frac, max_angle_deg):
    backup = path.replace('.csv', '.orig.csv')
    if os.path.exists(backup):
        df = pd.read_csv(backup)
    else:
        df = pd.read_csv(path)
        shutil.copy(path, backup)
    n0 = df.bird_id.nunique()

    span, disps = {}, {}
    for bid, g in df.groupby('bird_id'):
        g = g.sort_values('frame')
        span[bid] = (g.frame.iloc[0], g.frame.iloc[-1])
        disps[bid] = np.array([g.x.iloc[-1] - g.x.iloc[0], g.y.iloc[-1] - g.y.iloc[0]])

    long_ids = [b for b in span if span[b][1] - span[b][0] >= min_frames]
    # displacement gate relative to what the flock actually travels in this clip
    if long_ids:
        med_disp = np.median([np.linalg.norm(disps[b]) for b in long_ids])
        passing = [b for b in long_ids if np.linalg.norm(disps[b]) >= min_disp_frac * med_disp]
    else:
        passing = []

    keep = set()
    for bid in passing:
        a0, a1 = span[bid]
        # reference = displacement-weighted mean direction of temporally overlapping tracks
        ref = np.zeros(2)
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

    df[df.bird_id.isin(keep)].sort_values(['bird_id', 'frame']).to_csv(path, index=False, float_format='%.2f')
    return n0, len(keep)

def csvs_from_flight(tag):
    pattern = os.path.join(CSV_DIR, f"tracking_ts_{tag.strip()}_*.csv")
    return [f for f in sorted(glob.glob(pattern)) if not f.endswith('.orig.csv')]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csvs', nargs='*', help="Tracking CSVs to filter")
    parser.add_argument('--flights', type=str, help="File with {date}_Spot{n}_clip_{xxx} tags, both cameras")
    parser.add_argument('--min_frames', type=int, default=100, help="Minimum track duration in frames")
    parser.add_argument('--min_disp_frac', type=float, default=0.2, help="Min displacement as a fraction of the clip median")
    parser.add_argument('--max_angle', type=float, default=60.0, help="Max angle (deg) to the local flock direction")
    args = parser.parse_args()

    files = list(args.csvs)
    if args.flights:
        for tag in open(args.flights):
            if not tag.strip(): continue
            found = csvs_from_flight(tag)
            if not found:
                print(f"[WARN] no CSVs for flight {tag.strip()}")
            files.extend(found)

    if not files:
        print("Nothing to filter."); return
    total0, total1 = 0, 0
    for f in files:
        n0, n1 = filter_csv(f, args.min_frames, args.min_disp_frac, args.max_angle)
        total0 += n0; total1 += n1
        print(f"{os.path.basename(f)}: {n0} -> {n1} tracks")
    print(f"\n=== {len(files)} files | {total0} -> {total1} tracks ({total0 - total1} removed) ===")

if __name__ == "__main__":
    main()
