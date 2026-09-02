import os
import shutil
import argparse
import numpy as np
import pandas as pd
import cv2
from metavision_core.event_io import EventsIterator
from naming import build_raw_tag
from bird_tracking import render_frame

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CLICK_RADIUS = 30

class TrackEditor:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        self.max_frame = int(self.df.frame.max())
        self.selected = None
        self.hover = None
        self.modified = False
        self.reindex()

    def reindex(self):
        self.by_frame = {f: g for f, g in self.df.groupby('frame')}
        self.tracks = {int(b): (g.frame.to_numpy(), g.x.to_numpy(), g.y.to_numpy())
                       for b, g in self.df.sort_values('frame').groupby('bird_id')}

    def ids(self):
        return sorted(self.df.bird_id.unique())

    def at_click(self, frame, x, y):
        g = self.by_frame.get(frame)
        if g is None: return None
        d = np.hypot(g.x - x, g.y - y)
        i = d.idxmin()
        return int(g.loc[i, 'bird_id']) if d[i] < CLICK_RADIUS else None

    def delete_track(self, bid, quiet=False):
        n = (self.df.bird_id == bid).sum()
        self.df = self.df[self.df.bird_id != bid]
        self.reindex(); self.modified = True
        if not quiet: print(f"--> Track {bid} deleted ({n} points)")

    def delete_tracks(self, bids):
        self.df = self.df[~self.df.bird_id.isin(bids)]
        self.reindex(); self.modified = True
        print(f"--> {len(bids)} tracks deleted")

    def delete_point(self, bid, frame):
        mask = (self.df.bird_id == bid) & (self.df.frame == frame)
        if mask.any():
            self.df = self.df[~mask]
            self.reindex(); self.modified = True
            print(f"--> Point of {bid} at frame {frame} deleted")

    def merge(self, src, dst):
        """Merge track src into dst (dst keeps id and color; on frame conflicts dst wins)."""
        if src == dst or src not in self.ids() or dst not in self.ids():
            print("Invalid ids"); return
        dst_rows = self.df[self.df.bird_id == dst]
        dst_frames = set(dst_rows.frame)
        color = dst_rows.iloc[0][['color_b', 'color_g', 'color_r']].values
        conflict = (self.df.bird_id == src) & (self.df.frame.isin(dst_frames))
        self.df = self.df[~conflict]
        m = self.df.bird_id == src
        self.df.loc[m, 'bird_id'] = dst
        self.df.loc[self.df.bird_id == dst, ['color_b', 'color_g', 'color_r']] = color
        self.reindex(); self.modified = True
        print(f"--> Track {src} merged into {dst} ({conflict.sum()} conflicting points dropped)")

    def save(self):
        backup = self.csv_path.replace('.csv', '.orig.csv')
        if not os.path.exists(backup):
            shutil.copy(self.csv_path, backup)
            print(f"Original backed up at {backup}")
        self.df.sort_values(['bird_id', 'frame']).to_csv(self.csv_path, index=False, float_format='%.2f')
        self.modified = False
        print(f"Saved {self.csv_path} ({self.df.bird_id.nunique()} tracks)")

def draw_trail(display, pts_xy, color, bands=4):
    """Polyline that fades with age: oldest segments dimmest, newest at full color."""
    n = len(pts_xy)
    if n < 2: return
    idx = np.linspace(0, n - 1, bands + 1).astype(int)
    for i in range(bands):
        seg = pts_xy[idx[i]:idx[i + 1] + 1]
        w = (i + 1) / bands
        c = tuple(int(ch * w) for ch in color)
        if len(seg) > 1:
            cv2.polylines(display, [seg.astype(np.int32)], False, c, 1, cv2.LINE_AA)

def trail_points(ed, bid, frame, trail_frames):
    fr, xs, ys = ed.tracks[bid]
    m = (fr >= frame - trail_frames) & (fr <= frame)
    return np.column_stack([xs[m], ys[m]])

def nearest_point(ed, bid, frame, win):
    fr, xs, ys = ed.tracks[bid]
    i = np.argmin(np.abs(fr - frame))
    if abs(fr[i] - frame) > win: return None
    return int(xs[i]), int(ys[i])

def visible_points(ed, frame, tol):
    """Position of each track that is alive at this frame (within tol frames)."""
    pts = {}
    for bid in ed.tracks:
        p = nearest_point(ed, bid, frame, tol)
        if p is not None:
            pts[bid] = p
    return pts

def selection_phase(ed, ensure_frame, cache, height, width, step_frames, trail_frames):
    """Sweep the clip every step_frames; drag rectangles around the birds to KEEP."""
    checkpoints = list(range(0, ed.max_frame + 1, step_frames))
    tol = 10  # only tracks alive at the stop are shown/selectable
    rects = {i: [] for i in range(len(checkpoints))}
    drag = {'start': None, 'cur': None}
    ci = 0

    def kept_ids():
        """A track marked at any stop is kept whole, from its very start."""
        kept = set()
        for i, rs in rects.items():
            if not rs: continue
            pts = visible_points(ed, checkpoints[i], tol)
            for r in rs:
                x1, x2 = sorted((r[0], r[2])); y1, y2 = sorted((r[1], r[3]))
                for bid, (px, py) in pts.items():
                    if x1 <= px <= x2 and y1 <= py <= y2:
                        kept.add(bid)
        return kept

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            drag['start'] = (x, y); drag['cur'] = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and drag['start'] is not None:
            drag['cur'] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and drag['start'] is not None:
            r = (*drag['start'], x, y)
            if abs(r[2] - r[0]) > 5 and abs(r[3] - r[1]) > 5:
                rects[ci].append(r)
            drag['start'] = None; drag['cur'] = None

    cv2.setMouseCallback("Track Editor", on_mouse)
    print("\n--- PHASE 1: AREA SELECTION ---")
    print(" [mouse drag] : box the birds you WANT to keep (multiple boxes per stop)")
    print(" [n]/[b] : next/previous stop    [u] : undo last box")
    print(" [k]     : finish selection and delete unmarked tracks")
    print(" [q]     : skip this phase without deleting")
    print("-----------------------------------\n")

    while True:
        f = ensure_frame(checkpoints[ci])
        display = cache[f].copy()
        kept = kept_ids()
        pts = visible_points(ed, checkpoints[ci], tol)
        alive = len(pts)
        for bid, p in pts.items():
            color = (0, 200, 0) if bid in kept else (128, 128, 128)
            draw_trail(display, trail_points(ed, bid, checkpoints[ci], trail_frames), color)
            cv2.circle(display, p, 4, color, -1)
            cv2.putText(display, str(bid), (p[0] + 6, p[1] - 6), 0, 0.4, color, 1, cv2.LINE_AA)
        for r in rects[ci]:
            cv2.rectangle(display, (r[0], r[1]), (r[2], r[3]), (0, 200, 0), 1)
        if drag['start'] is not None and drag['cur'] is not None:
            cv2.rectangle(display, drag['start'], drag['cur'], (0, 255, 255), 1)
        info = (f"Stop {ci+1}/{len(checkpoints)} (frame {checkpoints[ci]}) | visible {alive} | "
                f"marked {len(kept)}/{len(ed.tracks)} | [n]ext [b]ack [u]ndo [k] apply [q] skip")
        cv2.putText(display, info, (10, height - 12), 0, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow("Track Editor", display)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('n'): ci = min(ci + 1, len(checkpoints) - 1)
        elif key == ord('b'): ci = max(ci - 1, 0)
        elif key == ord('u') and rects[ci]: rects[ci].pop()
        elif key == ord('q'):
            print("Selection skipped, nothing deleted."); return
        elif key == ord('k'):
            kept = kept_ids()
            drop = [b for b in ed.ids() if b not in kept]
            if not kept:
                print("No boxes drawn, nothing deleted."); return
            starts = ed.df.groupby('bird_id').frame.min()
            ends = ed.df.groupby('bird_id').frame.max()
            print(f"Keeping {len(kept)} tracks, deleting {len(drop)}:")
            for b in drop:
                print(f"    id {b}: frames {starts[b]}-{ends[b]}")
            print("(check the list: birds entering mid-clip only show up at late stops)")
            if input("Apply? (y/n): ").lower() == 'y':
                ed.delete_tracks(drop)
            return

def main():
    parser = argparse.ArgumentParser(description="Manual track editor: area selection sweep + fine merge/delete")
    parser.add_argument('raw_file', type=str, help="Raw clip the CSV was generated from")
    parser.add_argument('--mode', type=str, default='time_surface')
    parser.add_argument('--camera', type=str, default='left')
    parser.add_argument('--dt', type=int, default=5000)
    parser.add_argument('--csv', type=str, default=None, help="Tracking CSV (default: derived from the raw path)")
    parser.add_argument('--step', type=float, default=3.0, help="Seconds between selection stops")
    parser.add_argument('--trail', type=float, default=3.0, help="Seconds of decaying trail in the visualization")
    parser.add_argument('--no_select', action='store_true', help="Skip the area selection phase")
    args = parser.parse_args()

    filetype = "ts" if args.mode == 'time_surface' else "evf"
    csv_path = args.csv or os.path.join(SCRIPT_DIR, f"../csv/tracking_{filetype}_{build_raw_tag(args.raw_file)}_{args.camera}.csv")
    if not os.path.exists(csv_path):
        print(f"CSV not found: {csv_path}"); return
    ed = TrackEditor(csv_path)
    print(f"{len(ed.df)} puntos, {ed.df.bird_id.nunique()} tracks, frames 0-{ed.max_frame}")

    mv_it = EventsIterator(input_path=args.raw_file, delta_t=args.dt)
    height, width = mv_it.get_size()
    ts_surface = np.zeros((height, width), dtype=np.uint64) if args.mode == 'time_surface' else None
    iterator = iter(mv_it)
    cache = {}
    state = {'frame': 0, 'last_cached': -1}

    def ensure_frame(idx):
        while state['last_cached'] < idx:
            try:
                evs = next(iterator)
            except StopIteration:
                return min(idx, state['last_cached'])
            state['last_cached'] += 1
            if len(evs['x']) > 0:
                cache[state['last_cached']] = render_frame(evs, args.mode, args.dt, ts_surface, height, width)
            else:
                cache[state['last_cached']] = cache.get(state['last_cached'] - 1, np.zeros((height, width, 3), np.uint8)).copy()
        return idx

    cv2.namedWindow("Track Editor")
    trail_frames = max(1, int(args.trail * 1e6 / args.dt))

    # phase 1: area selection
    if not args.no_select:
        step_frames = max(1, int(args.step * 1e6 / args.dt))
        selection_phase(ed, ensure_frame, cache, height, width, step_frames, trail_frames)

    # phase 2: fine editing (merge/delete)
    def on_mouse(event, x, y, flags, param):
        ed.hover = ed.at_click(state['frame'], x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            ed.selected = ed.hover
            if ed.selected is not None:
                print(f"Selected: {ed.selected}")
    cv2.setMouseCallback("Track Editor", on_mouse)

    print("\n--- PHASE 2: FINE EDITING ---")
    print(" [n]/[b] : frame +1/-1      [f]/[v] : frame +25/-25")
    print(" [click] : select a bird (shows its full track)")
    print(" [m]     : merge selected INTO another id (asks for target id in terminal)")
    print(" [x]     : delete whole track    [d] : delete point at this frame")
    print(" [s]     : save (creates .orig.csv backup the first time)")
    print(" [q]     : quit")
    print("----------------------------\n")

    frame_idx = 0
    while True:
        frame_idx = max(0, min(frame_idx, ed.max_frame))
        frame_idx = ensure_frame(frame_idx)
        state['frame'] = frame_idx
        display = cache[frame_idx].copy()

        g = ed.by_frame.get(frame_idx)
        if g is not None:
            for _, row in g.iterrows():
                bid = int(row.bird_id)
                color = (int(row.color_b), int(row.color_g), int(row.color_r))
                if bid == ed.selected: color = (0, 0, 255)
                elif bid == ed.hover: color = (0, 255, 255)
                x1, y1 = int(row.x - row.w / 2), int(row.y - row.h / 2)
                x2, y2 = int(row.x + row.w / 2), int(row.y + row.h / 2)
                draw_trail(display, trail_points(ed, bid, frame_idx, trail_frames), color)
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 1 + (bid == ed.selected))
                cv2.putText(display, str(bid), (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        if ed.selected is not None and ed.selected in ed.tracks:
            fr, xs, ys = ed.tracks[ed.selected]
            pts = np.column_stack([xs, ys]).astype(np.int32)
            if len(pts) > 1:
                cv2.polylines(display, [pts], False, (0, 0, 255), 1, cv2.LINE_AA)

        info = f"Frame {frame_idx}/{ed.max_frame} | {ed.df.bird_id.nunique()} tracks"
        if ed.selected is not None:
            info += f" | ID {ed.selected}: [m]erge [x] delete track [d] delete point"
        cv2.putText(display, info, (10, height - 12), 0, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        if ed.modified:
            cv2.putText(display, "UNSAVED (s)", (width - 220, 25), 0, 0.6, (0, 0, 255), 2)
        cv2.imshow("Track Editor", display)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('n'): frame_idx += 1
        elif key == ord('b'): frame_idx -= 1
        elif key == ord('f'): frame_idx += 25
        elif key == ord('v'): frame_idx -= 25
        elif key == ord('s'): ed.save()
        elif key == ord('q'):
            if ed.modified:
                print("Unsaved changes: press [s] to save or [q] again to quit without saving")
                ed.modified = False
            else:
                break
        elif ed.selected is not None:
            if key == ord('m'):
                try:
                    dst = input(f"Merge {ed.selected} into id: ")
                    if dst:
                        ed.merge(ed.selected, int(dst))
                        ed.selected = int(dst)
                except ValueError:
                    print("Invalid id")
            elif key == ord('x'):
                if input(f"Delete whole track {ed.selected}? (y/n): ").lower() == 'y':
                    ed.delete_track(ed.selected)
                    ed.selected = None
            elif key == ord('d'):
                ed.delete_point(ed.selected, frame_idx)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
