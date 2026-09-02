import os
import cv2
import csv
import math
import numpy as np
import argparse
from ultralytics import YOLO
from metavision_core.event_io import EventsIterator
from naming import build_raw_tag

# CONFIGURATION
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH_TS = os.path.join(SCRIPT_DIR, "../config/yolo/ts.pt")
MODEL_PATH_EVF = os.path.join(SCRIPT_DIR, "../config/yolo/evf.pt")
CONFIDENCE = 0.5
LOW_CONFIDENCE = 0.1  # detections in [LOW, CONFIDENCE) may only continue tracks (ByteTrack-style)
VALIDATION_COUNT = 3
MAX_GAP_FRAMES = 15
BORDER_MARGIN = 25 
VELOCITY_SMOOTHING = 0.1 
MIN_TOTAL_DURATION = 15
FALLBACK_MEDIAN_VELOCITY = 1.73  # px/frame, used only if the pre-pass cannot estimate it
VELOCITY_TOLERANCE_PCT = 400
HARD_MAX_DIST = 50

# Colorblind-safe categorical palette (hex), cycled per bird id
TRACK_PALETTE_HEX = ["#0173b2", "#de8f05", "#029e73", "#d55e00", "#cc78bc", "#56b4e9"]

def hex_to_bgr(h):
    h = h.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)

TRACK_PALETTE_BGR = [hex_to_bgr(h) for h in TRACK_PALETTE_HEX]

class AssociationTracker:
    def __init__(self, min_hits, width, height, max_step, age_max=3, gap_max=MAX_GAP_FRAMES,
                 hard_max=HARD_MAX_DIST, bridge_rate=None, vel_smoothing=VELOCITY_SMOOTHING,
                 min_total=MIN_TOTAL_DURATION, conf_high=CONFIDENCE, wing_weight=0.0, dt_s=0.005):
        self.tracks = []
        self.history = []
        self.min_hits = min_hits
        self.width = width
        self.height = height
        self.max_step = max_step
        self.age_max = age_max
        self.gap_max = gap_max
        self.hard_max = hard_max
        self.bridge_rate = bridge_rate if bridge_rate is not None else max_step  # px per gap frame
        self.vel_smoothing = vel_smoothing
        self.min_total = min_total
        self.conf_high = conf_high
        self.wing_weight = wing_weight
        self.wing_band = (2.0 * dt_s, 12.0 * dt_s)  # 2-12 Hz in cycles/frame
        self.id_count = 0
        self.links = set()  # (det_key, det_key) pairs of matched transitions, for FB consensus
        
    def is_near_border(self, center):
        cx, cy = center
        return (cx < BORDER_MARGIN or cx > self.width - BORDER_MARGIN or 
                cy < BORDER_MARGIN or cy > self.height - BORDER_MARGIN)

    @staticmethod
    def _greedy_assign(preds, cents, max_step):
        """Global greedy: closest (track, detection) pairs win first."""
        assignment = {}
        if len(preds) == 0 or len(cents) == 0: return assignment
        dmat = np.linalg.norm(preds[:, None, :] - cents[None, :, :], axis=2)
        dmat[dmat > max_step] = np.inf
        order = np.argsort(dmat, axis=None)
        used_t, used_d = set(), set()
        n_d = len(cents)
        for flat in order:
            if not np.isfinite(dmat.flat[flat]): break
            ti, di = divmod(int(flat), n_d)
            if ti in used_t or di in used_d: continue
            assignment[ti] = di
            used_t.add(ti); used_d.add(di)
        return assignment

    def _apply_match(self, track, det, frame_idx):
        new_center = np.array(det['center'])
        if self.is_near_border(new_center):
            self.history.append(track)
            return False
        inst_vel = new_center - np.array(track['center'])
        track['smooth_vel'] = (self.vel_smoothing * inst_vel) + ((1 - self.vel_smoothing) * track['smooth_vel'])
        track['center'] = tuple(new_center)
        track['bbox'] = det['bbox']
        track['path'].append((tuple(new_center), frame_idx, det['size']))
        track['hits'] += 1
        track['age'] = 0
        if det.get('key') is not None:
            if track.get('last_key') is not None:
                self.links.add((track['last_key'], det['key']))
            track['last_key'] = det['key']
        return True

    def update(self, detections, frame_idx):
        def prep(items):
            out = []
            for item in items:
                x1, y1, x2, y2 = item['bbox']
                out.append({'center': item['center'], 'bbox': item['bbox'], 'size': (x2 - x1, y2 - y1),
                            'key': item.get('key'), 'matched': False})
            return out

        # ByteTrack-style two stages: confident detections start and continue tracks,
        # low-confidence ones (conf < conf_high) may only CONTINUE existing tracks
        high = prep([d for d in detections if d.get('conf', 1.0) >= self.conf_high])
        low = prep([d for d in detections if d.get('conf', 1.0) < self.conf_high])

        preds = np.array([np.array(t['center']) + t['smooth_vel'] for t in self.tracks]) if self.tracks else np.zeros((0, 2))
        cents_h = np.array([d['center'] for d in high], dtype=float) if high else np.zeros((0, 2))
        assign_h = self._greedy_assign(preds, cents_h, self.max_step)

        new_tracks = []
        unmatched = []
        for ti, track in enumerate(self.tracks):
            di = assign_h.get(ti)
            if di is not None:
                high[di]['matched'] = True
                if self._apply_match(track, high[di], frame_idx):
                    new_tracks.append(track)
            else:
                unmatched.append(track)

        # leftover tracks try the low-confidence detections
        if unmatched and low:
            preds_u = np.array([np.array(t['center']) + t['smooth_vel'] for t in unmatched])
            cents_l = np.array([d['center'] for d in low], dtype=float)
            assign_l = self._greedy_assign(preds_u, cents_l, self.max_step)
        else:
            assign_l = {}
        for ti, track in enumerate(unmatched):
            di = assign_l.get(ti)
            if di is not None:
                low[di]['matched'] = True
                if self._apply_match(track, low[di], frame_idx):
                    new_tracks.append(track)
            else:
                track['age'] += 1
                track['center'] = tuple(np.array(track['center']) + track['smooth_vel'])
                if track['age'] <= self.age_max:
                    new_tracks.append(track)
                else:
                    self.history.append(track)

        for det in high:
            if not det['matched'] and not self.is_near_border(det['center']):
                self.id_count += 1
                bird_color = TRACK_PALETTE_BGR[(self.id_count - 1) % len(TRACK_PALETTE_BGR)]
                new_tracks.append({
                    'id': self.id_count, 'center': det['center'], 'bbox': det['bbox'],
                    'smooth_vel': np.array([0.0, 0.0]),
                    'path': [(det['center'], frame_idx, det['size'])],
                    'hits': 1, 'age': 0, 'color': bird_color, 'last_key': det.get('key')
                })
        self.tracks = new_tracks
        return [t for t in self.tracks if t['hits'] >= self.min_hits]

    def _split_jumpy(self, fragments):
        """Cut fragments at internal jumps (>4x the fragment's median speed) so wrong
        stitches from the online pass become separate pieces for re-bridging."""
        out = []
        for t in fragments:
            path = t['path']
            if len(path) < 3:
                out.append(t); continue
            p = np.array([pt[0] for pt in path])
            fr = np.array([pt[1] for pt in path])
            step = np.linalg.norm(np.diff(p, axis=0), axis=1) / np.maximum(1, np.diff(fr))
            med = max(0.5, float(np.median(step)))
            cuts = np.where(step > 4 * med)[0]
            if len(cuts) == 0:
                out.append(t); continue
            idxs = [0] + [int(c) + 1 for c in cuts] + [len(path)]
            for k, (a, b) in enumerate(zip(idxs[:-1], idxs[1:])):
                seg = path[a:b]
                if len(seg) < 2: continue
                if k == 0:
                    seg_id, color = t['id'], t['color']
                else:
                    self.id_count += 1
                    seg_id = self.id_count
                    color = TRACK_PALETTE_BGR[(seg_id - 1) % len(TRACK_PALETTE_BGR)]
                sp = np.array([pt[0] for pt in seg[-5:]])
                sf = np.array([pt[1] for pt in seg[-5:]])
                vel = (sp[-1] - sp[0]) / max(1, sf[-1] - sf[0]) if len(sp) > 1 else np.zeros(2)
                out.append({'id': seg_id, 'path': seg, 'smooth_vel': vel, 'color': color,
                            'center': seg[-1][0], 'bbox': t['bbox'], 'hits': len(seg), 'age': 0})
        return out

    def _wing_sig(self, path, head, n=120):
        """Wingbeat signature (dominant frequency, amplitude) of the vertical oscillation
        at one end of a fragment. None if too short or too flat to be reliable."""
        pts = path[:n] if head else path[-n:]
        if len(pts) < 50: return None
        y = np.array([p[0][1] for p in pts], dtype=float)
        k = 31
        pad = np.pad(y, (k // 2, k // 2), mode='edge')
        osc = y - np.convolve(pad, np.ones(k) / k, mode='valid')[:len(y)]
        amp = osc.std()
        if amp < 0.15: return None
        spec = np.abs(np.fft.rfft(osc))
        freqs = np.fft.rfftfreq(len(osc), 1.0)  # cycles per frame
        m = (freqs >= self.wing_band[0]) & (freqs <= self.wing_band[1])
        if not m.any(): return None
        return float(freqs[m][np.argmax(spec[m])]), float(amp)

    def _motion_samples(self, fragments):
        """Per-frame samples (x, y, vx, vy) of the collective motion, from fragment steps."""
        samples = {}
        for t in fragments:
            path = t['path']
            for a, b in zip(path[:-1], path[1:]):
                df = b[1] - a[1]
                if df <= 0 or df > 3: continue
                samples.setdefault(b[1], []).append(
                    (b[0][0], b[0][1], (b[0][0] - a[0][0]) / df, (b[0][1] - a[0][1]) / df))
        return {f: np.array(v) for f, v in samples.items()}

    @staticmethod
    def _end_stats(path, head, k=5):
        pts = path[:k + 1] if head else path[-(k + 1):]
        p = np.array([pt[0] for pt in pts], dtype=float)
        fr = np.array([pt[1] for pt in pts])
        vel = (p[-1] - p[0]) / max(1, fr[-1] - fr[0]) if len(p) > 1 else np.zeros(2)
        full = np.array([pt[0] for pt in path], dtype=float)
        ffr = np.array([pt[1] for pt in path])
        if len(full) > 1:
            step = np.linalg.norm(np.diff(full, axis=0), axis=1) / np.maximum(1, np.diff(ffr))
            med = max(0.5, float(np.median(step)))
        else:
            med = 0.5
        return vel, med

    def bridge_trajectories(self):
        print("Starting trajectory bridging process (flock motion field)...")
        frags = [t for t in (self.history + self.tracks) if len(t['path']) >= self.min_hits]
        frags = self._split_jumpy(frags)
        samples = self._motion_samples(frags)
        frags.sort(key=lambda x: x['path'][0][1])
        starts = np.array([f['path'][0][1] for f in frags])
        tail_stats = [self._end_stats(f['path'], head=False) for f in frags]
        head_stats = [self._end_stats(f['path'], head=True) for f in frags]

        window_cache = {}
        def field(frame, pos, radius=200.0, win=6):
            """Local flock velocity around pos at frame (weighted by distance)."""
            if frame not in window_cache:
                pts = [samples[f] for f in range(frame - win, frame + win + 1) if f in samples]
                window_cache[frame] = np.concatenate(pts) if pts else None
            arr = window_cache[frame]
            if arr is None: return None
            d = np.hypot(arr[:, 0] - pos[0], arr[:, 1] - pos[1])
            m = d < radius
            if not m.any(): return None
            w = 1.0 / (1.0 + d[m] / 50.0)
            return np.array([np.average(arr[m, 2], weights=w), np.average(arr[m, 3], weights=w)])

        tail_sigs, head_sigs = {}, {}
        def sig(idx, head):
            store = head_sigs if head else tail_sigs
            if idx not in store:
                store[idx] = self._wing_sig(frags[idx]['path'], head)
            return store[idx]

        def integrate(pos, frame, fallback, forward):
            # Field-integrated prediction, curving with the flock instead of a straight line.
            preds = {}
            p = np.array(pos, dtype=float)
            for g in range(1, self.gap_max + 1):
                f = frame + g if forward else frame - g
                v = field(f, p)
                if v is None: v = np.asarray(fallback, dtype=float)
                p = p + v if forward else p - v
                preds[g] = p.copy()
            return preds

        bwd_cache = {}
        links = []
        for i, cur in enumerate(frags):
            end_pos = np.array(cur['path'][-1][0], dtype=float)
            end_f = cur['path'][-1][1]
            if self.is_near_border(end_pos): continue
            fwd = integrate(end_pos, end_f, cur['smooth_vel'], forward=True)
            vel_i, med_i = tail_stats[i]
            for j in range(int(np.searchsorted(starts, end_f + 1)), len(frags)):
                gap = frags[j]['path'][0][1] - end_f
                if gap > self.gap_max: break
                if gap <= 0 or j == i: continue
                sp = np.array(frags[j]['path'][0][0])
                seam = np.linalg.norm(sp - end_pos)
                if seam > min(self.hard_max, self.bridge_rate * gap): continue
                vel_j, med_j = head_stats[j]
                # Reject links that would create a measurable jump at the seam.
                if seam / gap > 4 * max(0.5, 0.5 * (med_i + med_j)): continue
                if j not in bwd_cache:
                    bwd_cache[j] = integrate(frags[j]['path'][0][0], frags[j]['path'][0][1], vel_j, forward=False)
                # Symmetric prediction error plus seam velocity continuity.
                err_f = np.linalg.norm(fwd[gap] - sp)
                err_b = np.linalg.norm(bwd_cache[j][gap] - end_pos)
                err = 0.5 * (err_f + err_b) / gap
                v_seam = (sp - end_pos) / gap
                err += 0.3 * 0.5 * (np.linalg.norm(v_seam - vel_i) + np.linalg.norm(v_seam - vel_j))
                if err >= self.max_step: continue
                if self.wing_weight > 0:
                    st, sh = sig(i, head=False), sig(j, head=True)
                    if st is not None and sh is not None:
                        mismatch = min(2.0, abs(np.log(st[0] / sh[0])) + 0.5 * abs(np.log(st[1] / sh[1])))
                        err += self.wing_weight * self.max_step * 0.5 * mismatch
                if err < self.max_step:
                    links.append((err, i, j))
        links.sort(key=lambda x: x[0])
        next_of, prev_of = {}, {}
        for err, i, j in links:
            if i in next_of or j in prev_of: continue
            next_of[i] = j; prev_of[j] = i

        merged = []
        for i, f in enumerate(frags):
            if i in prev_of: continue
            k = i
            while k in next_of:
                k = next_of[k]
                f['path'].extend(frags[k]['path'])
            merged.append(f)

        return [t for t in merged if len(t['path']) >= self.min_total]

def track_offline(frames_dets, width, height, tracker_kwargs, min_hits=VALIDATION_COUNT):
    """Forward-backward consensus tracking over stored detections.

    Runs the online associator in both time directions and keeps only the
    detection-to-detection links both passes agree on (ambiguous crossings become
    honest fragment breaks); the flock-field bridging then re-joins the fragments."""
    def run(reverse):
        tr = AssociationTracker(min_hits=min_hits, width=width, height=height, **tracker_kwargs)
        for k, f in enumerate(sorted(frames_dets, reverse=reverse)):
            tr.update(frames_dets[f], k)
        return {tuple(sorted(l, key=lambda key: key[0])) for l in tr.links}

    consensus = run(False) & run(True)

    nxt = {}
    prev = {}
    for a, b in consensus:
        if a in nxt or b in prev: continue
        nxt[a] = b; prev[b] = a

    det_by_key = {d['key']: d for f in frames_dets for d in frames_dets[f]}
    fragments = []
    idc = 0
    for a in nxt:
        if a in prev: continue
        chain = [a]
        k = a
        while k in nxt:
            k = nxt[k]
            chain.append(k)
        if len(chain) < min_hits: continue
        path = []
        for key in chain:
            d = det_by_key[key]
            x1, y1, x2, y2 = d['bbox']
            path.append((tuple(d['center']), key[0], (x2 - x1, y2 - y1)))
        idc += 1
        p = np.array([pt[0] for pt in path[-5:]])
        fr = np.array([pt[1] for pt in path[-5:]])
        vel = (p[-1] - p[0]) / max(1, fr[-1] - fr[0]) if len(p) > 1 else np.zeros(2)
        fragments.append({'id': idc, 'path': path, 'smooth_vel': vel,
                          'color': TRACK_PALETTE_BGR[(idc - 1) % len(TRACK_PALETTE_BGR)],
                          'center': path[-1][0], 'bbox': det_by_key[chain[-1]]['bbox'],
                          'hits': len(path), 'age': 0})

    bridger = AssociationTracker(min_hits=min_hits, width=width, height=height, **tracker_kwargs)
    bridger.history = fragments
    bridger.id_count = idc
    return bridger.bridge_trajectories()

def save_trajectories_to_csv(final_trajs, filename="bird_tracking_data.csv"):
    header = ['bird_id', 'frame', 'x', 'y', 'w', 'h', 'velocity_px_frame', 'heading_degrees', 'color_b', 'color_g', 'color_r']
    with open(filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for t in final_trajs:
            bird_id = t['id']
            path = t['path']
            c = t.get('color', (0, 0, 255)) 
            for i in range(len(path)):
                pos, frame, size = path[i]
                x, y = pos
                w, h = size
                
                vel = 0.0
                angle = 0.0
                if i > 0:
                    prev_pos = path[i-1][0]
                    dx = x - prev_pos[0]
                    dy = y - prev_pos[1]
                    vel = math.sqrt(dx**2 + dy**2)
                    angle = math.degrees(math.atan2(-dy, dx))
                writer.writerow([bird_id, frame, round(x, 2), round(y, 2), round(w, 2), round(h, 2), round(vel, 2), round(angle, 2), c[0], c[1], c[2]])
    print(f"--- Data exported successfully to {filename} ---")

def render_frame(evs, mode, dt, ts_surface, height, width):
    if mode == 'time_surface':
        x, y, t = evs['x'], evs['y'], evs['t']
        ts_surface[y, x] = t
        intensity = 255 * (1.0 - (np.clip(t[-1] - ts_surface, 0, dt) / dt))
        return cv2.cvtColor(intensity.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    event_frame = np.full((height, width, 3), 255, dtype=np.uint8)
    x, y, p = evs['x'], evs['y'], evs['p']
    event_frame[y[p == 1], x[p == 1]] = [255, 0, 0]
    event_frame[y[p == 0], x[p == 0]] = [0, 0, 255]
    return event_frame

def estimate_median_velocity(event_file_path, mode, dt, model, height, width,
                             min_samples=150, max_frames=300):
    """Pre-pass over the clip start: median frame-to-frame nearest-neighbour displacement
    of YOLO detections, in px/frame. Returns None if there is not enough data."""
    mv_it = EventsIterator(input_path=event_file_path, delta_t=dt)
    ts_surface = np.zeros((height, width), dtype=np.uint64) if mode == 'time_surface' else None
    prev_centers = None
    steps = []
    frame_count = 0
    for evs in mv_it:
        if len(evs['x']) == 0: continue
        display_bgr = render_frame(evs, mode, dt, ts_surface, height, width)
        results = model.predict(display_bgr, imgsz=1024, conf=CONFIDENCE, verbose=False)
        centers = []
        for result in results:
            if result.boxes is None: continue
            for box in result.boxes.xyxy.cpu().numpy():
                centers.append(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2))
        centers = np.array(centers)
        if prev_centers is not None and len(prev_centers) > 0 and len(centers) > 0:
            dists = np.linalg.norm(centers[:, None, :] - prev_centers[None, :, :], axis=2)
            steps.extend(dists.min(axis=1).tolist())
        prev_centers = centers
        frame_count += 1
        if len(steps) >= min_samples or frame_count >= max_frames:
            break
    if len(steps) < 10:
        return None
    return float(np.clip(np.median(steps), 0.5, 30.0))

def draw_tracks(frame, tracks):
    vis = frame.copy()
    for t in tracks:
        color = t['color']
        pts = np.array([p[0] for p in t['path']], dtype=np.int32)
        if len(pts) > 1:
            cv2.polylines(vis, [pts], isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA)
        x1, y1, x2, y2 = map(int, t['bbox'])
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis, f"ID {t['id']}", (x1, max(12, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return vis

def save_trajectory_plot(final_trajs, filename):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['CMU Serif', 'Computer Modern Roman', 'DejaVu Serif'],
        'mathtext.fontset': 'cm',
        'font.size': 16,
    })

    fig, ax = plt.subplots(figsize=(8, 6))
    handles = []
    for t in final_trajs:
        b, g, r = t.get('color', (178, 115, 1))
        color = (r / 255.0, g / 255.0, b / 255.0)
        xs = [p[0][0] for p in t['path']]
        ys = [p[0][1] for p in t['path']]
        ax.plot(xs, ys, color=color, linewidth=3, solid_capstyle='round')
        handles.append(Line2D([0], [0], color=color, linewidth=8, label=f"Bird {t['id']}"))

    ax.invert_yaxis()  # image coordinates: y grows downward
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, linestyle='--', linewidth=0.5, color='0.85')
    ax.tick_params(labelbottom=False, labelleft=False, length=0)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    if handles:
        ax.legend(handles=handles, loc='upper right', framealpha=1.0,
                  edgecolor='black', fancybox=True, handlelength=1.5)
    fig.savefig(filename, bbox_inches='tight')
    plt.close(fig)
    print(f"--- Trajectory plot saved to {filename} ---")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('raw_file', type=str, help="Choose .raw clip to process")
    parser.add_argument('--mode', type=str, default='time_surface', help="Mode of processing: 'time_surface' or 'event_frame'")
    parser.add_argument('--camera', type=str, default='left', help="Processing camera: 'left' or 'right'")
    parser.add_argument('--dt', type=int, default=10000, help="Delta time in microseconds")
    parser.add_argument('--save_csv', type=str, default='true')
    parser.add_argument('--vis', action='store_true', help="Show live tracking visualization per bird")
    parser.add_argument('--save_video', action='store_true', help="Save the tracking visualization as a video")
    parser.add_argument('--save_plot', action='store_true', help="Save a PDF plot of the final MOE trajectories")

    args = parser.parse_args()
    save_csv_bool = args.save_csv.lower() == 'true'

    if args.mode not in ['time_surface', 'event_frame']:
        print("Invalid mode choice. Use 'time_surface' or 'event_frame'.")
        return
    if args.camera not in ['left', 'right']:
        print("Invalid camera choice. Use 'left' or 'right'.")
        return

    event_file_path = args.raw_file
    raw_tag = build_raw_tag(event_file_path)
    dt = args.dt
    
    if args.mode == 'time_surface':
        filetype = "ts"
        model = YOLO(MODEL_PATH_TS)
    elif args.mode == 'event_frame':
        filetype = "evf"
        model = YOLO(MODEL_PATH_EVF)

    csv_dir = os.path.join(SCRIPT_DIR, "../csv")
    os.makedirs(csv_dir, exist_ok=True)
    filename = os.path.join(csv_dir, f"tracking_{filetype}_{raw_tag}_{args.camera}.csv")

    mv_it = EventsIterator(input_path=event_file_path, delta_t=dt)
    height, width = mv_it.get_size()

    print("Estimating median velocity from the clip...")
    median_vel = estimate_median_velocity(event_file_path, args.mode, dt, model, height, width)
    if median_vel is None:
        median_vel = FALLBACK_MEDIAN_VELOCITY
        print(f"Not enough detections to estimate velocity, using fallback {median_vel} px/frame")
    else:
        print(f"Estimated median velocity: {median_vel:.2f} px/frame")
    # Tuned gate parameters; factors scale the per-clip velocity
    params_file = os.path.join(SCRIPT_DIR, "../config/tracker_params.yaml")
    tp = {}
    if os.path.exists(params_file):
        import yaml
        tp = yaml.safe_load(open(params_file)) or {}
        print(f"Using tuned tracker params from {params_file}: {tp}")
    max_step = median_vel * tp.get('vel_tol_mult', 1 + VELOCITY_TOLERANCE_PCT / 100.0)

    tracker_kwargs = dict(
        max_step=max_step,
        age_max=tp.get('age_max', 3),
        gap_max=tp.get('gap_max', MAX_GAP_FRAMES),
        hard_max=tp.get('hard_max', HARD_MAX_DIST),
        bridge_rate=(median_vel * tp['bridge_rate_mult']) if 'bridge_rate_mult' in tp else None,
        vel_smoothing=tp.get('vel_smoothing', VELOCITY_SMOOTHING),
        min_total=tp.get('min_total', MIN_TOTAL_DURATION),
        conf_high=CONFIDENCE,
        wing_weight=tp.get('wing_weight', 0.3),
        dt_s=dt * 1e-6)
    if args.mode == 'time_surface':
        ts_surface = np.zeros((height, width), dtype=np.uint64)
    else:
        ts_surface = None

    video_writer = None
    if args.save_video:
        video_dir = os.path.join(SCRIPT_DIR, "../videos")
        os.makedirs(video_dir, exist_ok=True)
        video_path = os.path.join(video_dir, f"tracking_{filetype}_{raw_tag}_{args.camera}.mp4")
        fps = 1e6 / dt
        video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    frame_idx = 0
    frames_dets = {}

    print("Processing...")
    for evs in mv_it:
        if len(evs['x']) == 0: continue
        display_bgr = render_frame(evs, args.mode, dt, ts_surface, height, width)

        # detect down to LOW_CONFIDENCE: the tracker uses <CONFIDENCE ones only to continue tracks
        results = model.predict(display_bgr, imgsz=1024, conf=LOW_CONFIDENCE, verbose=False)
        detection_list = []

        for result in results:
            if result.boxes is None: continue

            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()

            if result.masks is not None:

                for box, conf in zip(boxes, confs):

                    x1, y1, x2, y2 = map(int, box)
                    
                    x1 = max(0, x1); y1 = max(0, y1)
                    x2 = min(width, x2); y2 = min(height, y2)
                    
                    roi = display_bgr[y1:y2, x1:x2]
                                  
                    mask = None
                
                    if roi.size == 0: continue 

                    if args.mode == 'time_surface':
                        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                        _, mask = cv2.threshold(gray_roi, 30, 255, cv2.THRESH_BINARY)
                        
                    elif args.mode == 'event_frame':
                        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                        _, mask = cv2.threshold(gray_roi, 250, 255, cv2.THRESH_BINARY_INV)
                    
                    cx, cy = 0.0, 0.0
                    
                    if mask is not None:
                        M = cv2.moments(mask)
                        if M['m00'] > 0:
                            cx_roi = M['m10'] / M['m00']
                            cy_roi = M['m01'] / M['m00']
                            
                            cx = x1 + cx_roi
                            cy = y1 + cy_roi

                    w = box[2] - box[0]
                    h = box[3] - box[1]
                    
                    detection_list.append({
                        'bbox': box,
                        'center': (cx, cy),
                        'size': (w, h),
                        'conf': float(conf)
                    })

            else:
                for box, conf in zip(boxes, confs):
                    cx = (box[0] + box[2]) / 2
                    cy = (box[1] + box[3]) / 2
                    detection_list.append({'bbox': box, 'center': (cx, cy), 'conf': float(conf)})

        for i, d in enumerate(detection_list):
            d['key'] = (frame_idx, i)
        frames_dets[frame_idx] = detection_list

        if args.vis or video_writer is not None:
            vis_frame = display_bgr.copy()
            for d in detection_list:
                x1, y1, x2, y2 = map(int, d['bbox'])
                color = (0, 200, 0) if d['conf'] >= CONFIDENCE else (128, 128, 128)
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 1)
            if video_writer is not None:
                video_writer.write(vis_frame)
            if args.vis:
                cv2.imshow("Bird detections (q to stop)", vis_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Stopped by user, tracking the frames seen so far...")
                    break

        frame_idx += 1

    if video_writer is not None:
        video_writer.release()
        print(f"--- Video saved to {video_path} ---")
    if args.vis:
        cv2.destroyAllWindows()

    print("Running forward-backward consensus tracking...")
    final_trajs = track_offline(frames_dets, width, height, tracker_kwargs)
    print(f"Post-processing complete. Found {len(final_trajs)} valid bird paths.")

    if save_csv_bool:
        save_trajectories_to_csv(final_trajs, filename)

    if args.save_plot:
        plot_dir = os.path.join(SCRIPT_DIR, "../plots")
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, f"trajectories_{filetype}_{raw_tag}_{args.camera}.pdf")
        save_trajectory_plot(final_trajs, plot_path)

if __name__ == "__main__":
    main()