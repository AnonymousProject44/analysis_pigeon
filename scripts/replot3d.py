import os
import io
import argparse
import contextlib
import subprocess
import pandas as pd

from filter_trajectories3d import regen_plot, CSV_DIR

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PNG_DIR = os.path.join(SCRIPT_DIR, "../plots/png")

def resolve(target):
    if os.path.exists(target):
        return target
    p = os.path.join(CSV_DIR, f"matching_{target}_ts.csv")
    return p if os.path.exists(p) else None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('targets', nargs='*', help="Flight tags or matching_*.csv paths")
    parser.add_argument('--flights', type=str, help="File with {date}_Spot{n}_clip_{xxx} tags")
    parser.add_argument('--dpi', type=int, default=150)
    args = parser.parse_args()

    tags = list(args.targets)
    if args.flights:
        tags += [l.strip() for l in open(args.flights) if l.strip()]

    os.makedirs(PNG_DIR, exist_ok=True)
    done = 0
    for t in tags:
        path = resolve(t)
        if path is None:
            print(f"[WARN] not found: {t}"); continue
        tag = os.path.basename(path).replace('matching_', '').replace('_ts.csv', '')
        with contextlib.redirect_stdout(io.StringIO()):
            regen_plot(path, pd.read_csv(path))
        pdf = os.path.join(SCRIPT_DIR, f"../plots/trajectories3d_{tag}_ts.pdf")
        subprocess.run(["pdftoppm", "-png", "-r", str(args.dpi), "-singlefile",
                        pdf, os.path.join(PNG_DIR, f"trajectories3d_{tag}_ts")], check=True)
        print(f"{tag}: pdf + png")
        done += 1
    print(f"\n=== {done} flights replotted ===")

if __name__ == "__main__":
    main()
