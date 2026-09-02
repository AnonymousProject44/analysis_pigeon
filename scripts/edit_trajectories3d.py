import os
import io
import shutil
import argparse
import contextlib
import numpy as np
import pandas as pd
import cv2

from filter_trajectories3d import positions, sync_companions, regen_plot
from bird_tracking import TRACK_PALETTE_BGR

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(SCRIPT_DIR, "../csv")
W, H = 1400, 900
CLICK_PX = 12

class Editor3D:
    def __init__(self, path):
        self.path = path
        self.df = pd.read_csv(path)
        self.undo = []
        self.selected = None
        self.merge_mode = False
        self.modified = False
        self.yaw, self.pitch = 0.6, 0.35
        self.pan = np.zeros(2)
        self.drag = None
        self.moved = 0.0
        self.show_outliers = False  # default view matches the PDF (filter_outliers applied)
        self.rebuild()
        self.zoom = 0.8 * min(W, H) / self.range0

    def rebuild(self):
        """Refresh per-trajectory arrays and the scene center from the dataframe."""
        if self.show_outliers:
            src = self.df
        else:
            from stereo_visualizer import filter_outliers
            with contextlib.redirect_stdout(io.StringIO()):
                src = filter_outliers(self.df)
        self.traj = {}
        pts_all = []
        for bid, g in src.groupby('bird_id'):
            g = g.sort_values('timestamp')
            if len(g) < 2: continue
            p = positions(g)
            self.traj[int(bid)] = np.column_stack([p[:, 0], -p[:, 1], p[:, 2]])  # x, height, depth
            pts_all.append(self.traj[int(bid)])
        pts = np.concatenate(pts_all) if pts_all else np.zeros((1, 3))
        lo, hi = np.percentile(pts, 2, axis=0), np.percentile(pts, 98, axis=0)
        self.center = (lo + hi) / 2
        self.range0 = max((hi - lo).max(), 1.0)

    def project(self, p3):
        cy, sy = np.cos(self.yaw), np.sin(self.yaw)
        cp, sp = np.cos(self.pitch), np.sin(self.pitch)
        d = p3 - self.center
        x = cy * d[:, 0] + sy * d[:, 2]
        z = -sy * d[:, 0] + cy * d[:, 2]
        y = cp * d[:, 1] - sp * z
        sx = W / 2 + self.zoom * x + self.pan[0]
        sy_ = H / 2 - self.zoom * y + self.pan[1]
        return np.column_stack([sx, sy_])

    def snapshot(self):
        self.undo.append(self.df.copy())
        if len(self.undo) > 20: self.undo.pop(0)

    def nearest(self, x, y):
        best = (CLICK_PX, None)
        for bid, p3 in self.traj.items():
            step = max(1, len(p3) // 300)
            s = self.project(p3[::step])
            d = np.hypot(s[:, 0] - x, s[:, 1] - y).min()
            if d < best[0]:
                best = (d, bid)
        return best[1]

    def apply_click(self, x, y):
        bid = self.nearest(x, y)
        if bid is None:
            self.selected = None
            self.merge_mode = False
            return
        if self.merge_mode and self.selected is not None and bid != self.selected:
            self.snapshot()
            dst_frames = set(self.df[self.df.bird_id == bid].frame)
            conflict = (self.df.bird_id == self.selected) & (self.df.frame.isin(dst_frames))
            self.df = self.df[~conflict]
            self.df.loc[self.df.bird_id == self.selected, 'bird_id'] = bid
            print(f"--> {self.selected} merged into {bid} ({conflict.sum()} conflicting rows dropped)")
            self.selected = bid
            self.merge_mode = False
            self.modified = True
            self.rebuild()
        else:
            self.selected = bid
            print(f"Selected: {bid} ({len(self.traj[bid])} points)")

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag = (x, y)
            self.moved = 0.0
        elif event == cv2.EVENT_MOUSEMOVE and self.drag is not None:
            dx, dy = x - self.drag[0], y - self.drag[1]
            self.moved += abs(dx) + abs(dy)
            if flags & cv2.EVENT_FLAG_SHIFTKEY:
                self.pan += (dx, dy)
            else:
                self.yaw += dx * 0.01
                self.pitch = np.clip(self.pitch + dy * 0.01, -1.5, 1.5)
            self.drag = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            if self.moved < 5:
                self.apply_click(x, y)
            self.drag = None
        elif event == cv2.EVENT_MOUSEWHEEL:
            self.zoom *= 1.15 if flags > 0 else 1 / 1.15

    def render(self):
        img = np.full((H, W, 3), 20, dtype=np.uint8)
        for bid, p3 in self.traj.items():
            s = self.project(p3).astype(np.int32)
            if bid == self.selected:
                color, thick = (0, 0, 255), 2
            else:
                color, thick = TRACK_PALETTE_BGR[(bid - 1) % len(TRACK_PALETTE_BGR)], 1
            cv2.polylines(img, [s], False, color, thick, cv2.LINE_AA)
            cv2.putText(img, str(bid), tuple(s[-1] + [4, -4]), 0, 0.35,
                        (0, 0, 255) if bid == self.selected else (160, 160, 160), 1, cv2.LINE_AA)
        state = f"{self.df.bird_id.nunique()} trajectories ({len(self.traj)} shown, outliers {'shown' if self.show_outliers else 'hidden [o]'})"
        if self.selected is not None:
            state += f" | selected {self.selected}" + (" | MERGE: click target" if self.merge_mode else " | [m]erge [x] delete")
        state += " | [u]ndo [s]ave [r]eset view [q]uit | drag=rotate shift+drag=pan wheel or +/-=zoom"
        cv2.putText(img, state, (10, H - 12), 0, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
        if self.modified:
            cv2.putText(img, "UNSAVED (s)", (W - 160, 25), 0, 0.6, (0, 0, 255), 2)
        return img

    def save(self):
        backup = self.path.replace('.csv', '.orig.csv')
        if not os.path.exists(backup):
            shutil.copy(self.path, backup)
            print(f"Original backed up at {backup}")
        self.df.to_csv(self.path, index=False)
        sync_companions(self.path, set(self.df.bird_id.unique()))
        with contextlib.redirect_stdout(io.StringIO()):
            regen_plot(self.path, self.df)
        self.modified = False
        print(f"Saved {self.path} ({self.df.bird_id.nunique()} trajectories, plot regenerated)")

    def run(self):
        cv2.namedWindow("3D Trajectory Editor")
        cv2.setMouseCallback("3D Trajectory Editor", self.on_mouse)
        while True:
            cv2.imshow("3D Trajectory Editor", self.render())
            key = cv2.waitKey(15) & 0xFF
            if key == ord('m') and self.selected is not None:
                self.merge_mode = True
            elif key == ord('x') and self.selected is not None:
                self.snapshot()
                n = (self.df.bird_id == self.selected).sum()
                self.df = self.df[self.df.bird_id != self.selected]
                print(f"--> Trajectory {self.selected} deleted ({n} points)")
                self.selected = None
                self.merge_mode = False
                self.modified = True
                self.rebuild()
            elif key == ord('u') and self.undo:
                self.df = self.undo.pop()
                self.selected = None
                self.merge_mode = False
                self.modified = True
                print("--> Undo")
                self.rebuild()
            elif key == ord('o'):
                self.show_outliers = not self.show_outliers
                self.rebuild()
            elif key in (ord('+'), ord('=')):
                self.zoom *= 1.15
            elif key == ord('-'):
                self.zoom /= 1.15
            elif key == ord('r'):
                self.yaw, self.pitch, self.pan = 0.6, 0.35, np.zeros(2)
                self.zoom = 0.8 * min(W, H) / self.range0
            elif key == ord('s'):
                self.save()
            elif key == ord('q'):
                if self.modified:
                    print("Unsaved changes: press [s] to save or [q] again to quit without saving")
                    self.modified = False
                else:
                    break
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('target', type=str, help="Flight tag ({date}_Spot{n}_clip_{xxx}) or matching_*.csv path")
    args = parser.parse_args()
    path = args.target
    if not os.path.exists(path):
        path = os.path.join(CSV_DIR, f"matching_{args.target}_ts.csv")
    if not os.path.exists(path):
        print(f"Not found: {path}"); return

    print("\n--- 3D TRAJECTORY EDITOR ---")
    print(" [click] : select trajectory (red)")
    print(" [m]     : merge mode, then click the target trajectory")
    print(" [x]     : delete selected    [u] : undo (last 20 ops)")
    print(" [s]     : save (+plot regen) [r] : reset view [q] : quit")
    print(" [o]     : toggle outlier points (hidden by default, same view as the PDF)")
    print(" drag = rotate | shift+drag = pan | wheel or +/- = zoom\n")
    Editor3D(path).run()

if __name__ == "__main__":
    main()
