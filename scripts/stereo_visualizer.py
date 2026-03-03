import argparse
import os
import sys
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from metavision_core.event_io import EventsIterator
import re  

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

class BirdKalmanFilter:
    def __init__(self, dt=0.005):
        self.kf = cv2.KalmanFilter(6, 3)
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], np.float32)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ], np.float32)
        self.kf.processNoiseCov = np.eye(6, dtype=np.float32) * 1e-3
        self.kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * 0.5
        self.kf.errorCovPost = np.eye(6, dtype=np.float32)
        self.initialized = False

    def filter_point(self, x, y, z):
        if not self.initialized:
            self.kf.statePost = np.array([[x], [y], [z], [0], [0], [0]], np.float32)
            self.initialized = True
            return x, y, z
        self.kf.predict()
        measurement = np.array([[np.float32(x)], [np.float32(y)], [np.float32(z)]])
        estimate = self.kf.correct(measurement).reshape(-1)
        return estimate[0], estimate[1], estimate[2]

def smooth_trajectories(df):
    smoothed_rows = []
    for bid in df['bird_id'].unique():
        bird_data = df[df['bird_id'] == bid].sort_values('timestamp')
        kf = BirdKalmanFilter(dt=0.005)
        for _, row in bird_data.iterrows():
            sx, sy, sz = kf.filter_point(row['x_m'], row['y_m'], row['z_m'])
            new_row = row.copy()
            new_row['x_m'], new_row['y_m'], new_row['z_m'] = sx, sy, sz
            smoothed_rows.append(new_row)
    return pd.DataFrame(smoothed_rows)

def filter_outliers(df):
    original_count = len(df)
    df = df[(df['z_m'] > 0.5) & (df['z_m'] < 300)]
    for col in ['x_m', 'y_m', 'z_m']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 2.0 * IQR 
        upper_bound = Q3 + 2.0 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    print(f"Filtered {original_count - len(df)} outlier points. Remaining: {len(df)}")
    return df

def get_unique_color(id_index):
    colors = [
        (0, 255, 255), (255, 0, 255), (0, 255, 0), 
        (255, 128, 0), (0, 128, 255), (128, 255, 0), 
        (0, 0, 255), (255, 255, 0), (255, 0, 127), (127, 0, 255)
    ]
    return colors[id_index % len(colors)]

def compute_time_surface(events, ts_surface, current_time, fade_time=30000):
    if events.size > 0:
        x, y, t = events['x'].astype(int), events['y'].astype(int), events['t']
        ts_surface[y, x] = t
    time_diff = current_time - ts_surface
    intensity = (1.0 - np.clip(time_diff, 0, fade_time) / fade_time)
    return (intensity * 255).astype(np.uint8)

def compute_event_frame(event_frame, evs, dt):
    x = evs['x']
    y = evs['y']
    p = evs['p']
    pos_mask = (p == 1)
    neg_mask = (p == 0)
    event_frame[y[pos_mask], x[pos_mask]] = [255, 0, 0]
    event_frame[y[neg_mask], x[neg_mask]] = [0, 0, 255]
    return event_frame

def load_valid_ids(config_path):
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        return None
    try:
        with open(config_path, 'r') as f:
            content = f.read()
            match = re.search(r'available:\s*\[(.*?)\]', content)
            if match:
                ids_str = match.group(1)
                valid_ids = [int(x.strip()) for x in ids_str.split(',') if x.strip().isdigit()]
                print(f"Loaded {len(valid_ids)} valid IDs from config.")
                return valid_ids
            else:
                print("Could not find 'available: [...]' list in config.")
                return None
    except Exception as e:
        print(f"Error reading config: {e}")
        return None

def visualize_stereo_matches(iter_l, iter_r, df_l, df_r, df_stereo, df_matches, mode='event_frame',
                             width=1280, height=720, frame_dt=5000, save_video=False, output_file="stereo_output.mp4"):
    ts_l = np.zeros((height, width), dtype=np.uint64)
    ts_r = np.zeros((height, width), dtype=np.uint64)
    
    id_map = {}
    for i, row in df_matches.iterrows():
        left_id, right_id = int(row['Left_ID']), int(row['Right_ID'])
        id_map[left_id] = {"right_id": right_id, "stereo_id": i + 1, "color": get_unique_color(i)}
    
    depth_map = {}
    if not df_stereo.empty:
        depth_map = df_stereo.set_index(['bird_id', 'frame'])['z_m'].to_dict()

    print("Controls: [Space] Play/Pause | [q] Quit")
    paused = False
    data_stream = zip(iter_l, iter_r)
    video_writer = None

    while True:
        if not paused:
            try:
                ev_l, ev_r = next(data_stream)
            except StopIteration:
                break
            t_now = max(ev_l['t'][-1] if ev_l.size > 0 else 0, ev_r['t'][-1] if ev_r.size > 0 else 0)
            current_frame = int(t_now // frame_dt)
            if mode == 'event_frame':
                event_frame_l = np.zeros((height, width, 3), dtype=np.uint8)
                event_frame_r = np.zeros((height, width, 3), dtype=np.uint8)
                img_l = compute_event_frame(event_frame_l, ev_l, t_now)
                img_r = compute_event_frame(event_frame_r, ev_r, t_now)
            else:
                img_l = cv2.cvtColor(compute_time_surface(ev_l, ts_l, t_now), cv2.COLOR_GRAY2BGR)
                img_r = cv2.cvtColor(compute_time_surface(ev_r, ts_r, t_now), cv2.COLOR_GRAY2BGR)

            combined = np.hstack((img_l, img_r))
            
            tracks_r_now = df_r[df_r['frame'] == current_frame]
            centers_r = {int(row['bird_id']): (int(row['x']), int(row['y'])) for _, row in tracks_r_now.iterrows()}
            tracks_l_now = df_l[df_l['frame'] == current_frame]
            
            for _, row in tracks_l_now.iterrows():
                bid_l = int(row['bird_id'])
                cx_l, cy_l = int(row['x']), int(row['y'])
                if bid_l in id_map:
                    match_info = id_map[bid_l]
                    bid_r, pair_color = match_info["right_id"], match_info["color"]
                    if bid_r in centers_r:
                        cx_r, cy_r = centers_r[bid_r]
                        cx_r_shifted = cx_r + width
                        z_val = depth_map.get((bid_l, current_frame), 0.0)
                        cv2.circle(combined, (cx_l, cy_l), 7, pair_color, -1)
                        cv2.circle(combined, (cx_r_shifted, cy_r), 7, pair_color, -1)
                        label = f"ID{bid_l}-{bid_r} Z:{z_val:.1f}m"
                        cv2.putText(combined, label, (cx_l + 12, cy_l), cv2.FONT_HERSHEY_SIMPLEX, 0.5, pair_color, 1)
                        cv2.putText(combined, label, (cx_r_shifted + 12, cy_r), cv2.FONT_HERSHEY_SIMPLEX, 0.5, pair_color, 1)
                else:
                    cv2.circle(combined, (cx_l, cy_l), 4, (150, 150, 150), 1)

            display = cv2.resize(combined, (0, 0), fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)
            if save_video:
                if video_writer is None:
                    h, w = display.shape[:2]
                    video_writer = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (w, h))
                video_writer.write(display)
            if paused:
                cv2.putText(display, "PAUSED", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Stereo Tracking", display)
        key = cv2.waitKey(1 if not paused else 0) & 0xFF
        if key == ord('q'): break
        elif key == ord(' '): paused = not paused

    if video_writer: video_writer.release()
    cv2.destroyAllWindows()

def visualize_3d_trajectories(df_full):
    try:
        if df_full.empty:
            print("DataFrame is empty (after filtering).")
            return

        unique_ids = sorted(df_full['bird_id'].unique().astype(int))
        print(f"Plotting Bird IDs: {unique_ids}")

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        state = {'input_buffer': [], 'current_filter': None}
        t_min, t_max = df_full['timestamp'].min(), df_full['timestamp'].max()
        norm = mcolors.Normalize(vmin=t_min, vmax=t_max)
        cmap = cm.turbo

        mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array([])
        fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=15, pad=0.1, label='Time')

        def draw_plot():
            ax.clear()
            ax.set_xlabel('X (Lateral)')
            ax.set_ylabel('Z (Depth)')
            ax.set_zlabel('Y (Height)')

            if state['current_filter'] is not None:
                df = df_full[df_full['bird_id'] == state['current_filter']]
                title_extra = f" | Showing ID: {state['current_filter']}"
            else:
                df = df_full
                title_extra = " | Showing ALL Configured IDs"

            if not df.empty:
                for bid in df['bird_id'].unique():
                    bird_track = df[df['bird_id'] == bid].sort_values('timestamp')
                    if len(bird_track) > 1:
                        xs, zs, ys = bird_track['x_m'].values, bird_track['z_m'].values, -bird_track['y_m'].values
                        times = bird_track['timestamp'].values
                        points = np.array([xs, zs, ys]).T.reshape(-1, 1, 3)
                        segments = np.concatenate([points[:-1], points[1:]], axis=1)
                        lc = Line3DCollection(segments, cmap=cmap, norm=norm)
                        lc.set_array(times[:-1])
                        lc.set_linewidth(1.5)
                        lc.set_alpha(0.8)
                        ax.add_collection3d(lc)
                        ax.text(xs[0], zs[0], ys[0], f"ID {int(bid)}", fontsize=8, fontweight='bold', color='black')

                all_x, all_z, all_y = df['x_m'], df['z_m'], -df['y_m']
                mid_x, mid_z, mid_y = (all_x.max()+all_x.min())*0.5, (all_z.max()+all_z.min())*0.5, (all_y.max()+all_y.min())*0.5
                max_range = np.array([all_x.max()-all_x.min(), all_z.max()-all_z.min(), all_y.max()-all_y.min()]).max()/2.0
                ax.set_xlim(mid_x-max_range, mid_x+max_range)
                ax.set_ylim(mid_z-max_range, mid_z+max_range)
                ax.set_zlim(mid_y-max_range, mid_y+max_range)
                
            buffer_str = "".join(state['input_buffer'])
            ax.set_title(f'3D Trajectories{title_extra} | Typing: {buffer_str}\n[Enter=Update] [Esc=Quit]')
            fig.canvas.draw()

        def on_key(event):
            if event.key == 'escape': plt.close()
            elif event.key == 'enter':
                if state['input_buffer']:
                    try: state['current_filter'] = int("".join(state['input_buffer']))
                    except ValueError: pass
                    state['input_buffer'] = []
                else: state['current_filter'] = None
                draw_plot()
            elif event.key and event.key.isdigit():
                state['input_buffer'].append(event.key)
                draw_plot()

        fig.canvas.mpl_connect('key_press_event', on_key)
        draw_plot()
        plt.show()

    except Exception as e:
        print(f"Viz Error: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_l", type=str, default='clip_006')
    parser.add_argument('raw_r', type=str, default='clip_006')
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument('--clip', type=str, default='clip_006')
    parser.add_argument("--mode", type=str, default="event_frame")
    args = parser.parse_args()

    if args.mode not in ['time_surface', 'event_frame']:
        print("Invalid mode choice. Use 'time_surface' or 'event_frame'.")
        sys.exit(1)

    # Paths setup
    raw_l = args.raw_l
    raw_r = args.raw_r
    output_file = os.path.join(SCRIPT_DIR, f"../videos/stereo_output_{args.clip}.mp4")
    type = "evf" if args.mode == 'event_frame' else "ts"
    csv_l = os.path.join(SCRIPT_DIR, f'../csv/tracking_{type}_{args.clip}_left.csv')
    csv_r = os.path.join(SCRIPT_DIR, f'../csv/tracking_{type}_{args.clip}_right.csv')
    csv_stereo = os.path.join(SCRIPT_DIR, f'../csv/matching_{args.clip}_{type}.csv')
    csv_matches = os.path.join(SCRIPT_DIR,f'../csv/id_matches_{args.clip}_{type}.csv')

    try:
        if not os.path.exists(csv_stereo):
            print("Stereo CSV not found.")
            return

        df_l = pd.read_csv(csv_l)
        df_r = pd.read_csv(csv_r)
        df_stereo = pd.read_csv(csv_stereo)
        df_matches = pd.read_csv(csv_matches) if os.path.exists(csv_matches) else pd.DataFrame(columns=['Left_ID', 'Right_ID'])

        iter_l = EventsIterator(input_path=raw_l, delta_t=5000)
        iter_r = EventsIterator(input_path=raw_r, delta_t=5000)

        # Visual 2D 
        visualize_stereo_matches(iter_l, iter_r, df_l, df_r, df_stereo, df_matches, mode=args.mode, save_video=args.save_video, output_file=output_file)

        # Process Smoothing & Save CSV
        df_full = pd.DataFrame() 
        if not df_stereo.empty:
            print("Filtering and smoothing trajectories...")
            df_clean = filter_outliers(df_stereo)
            df_full = smooth_trajectories(df_clean)
            
            # Save ALL IDs (before filtering)
            df_export_all = df_full[['bird_id', 'frame', 'x_m', 'y_m', 'z_m']].copy()
            df_export_all.columns = ['bird_id', 'frame', 'x_m_smooth', 'y_m_smooth', 'z_m_smooth']
            save_path_all = f"csv/smoothed_bird_data_{args.clip}_{type}_all.csv"
            df_export_all.to_csv(save_path_all, index=False)
            print(f"Exported smoothed CSV (ALL IDs) to {save_path_all}")

            # Export requested CSV (Filtered)
            df_export = df_full[['bird_id', 'frame', 'x_m', 'y_m', 'z_m']].copy()
            df_export.columns = ['bird_id', 'frame', 'x_m_smooth', 'y_m_smooth', 'z_m_smooth']
            save_path = f"csv/smoothed_bird_data_{args.clip}_{type}.csv"
            df_export.to_csv(save_path, index=False)
            print(f"Exported smoothed CSV (Filtered) to {save_path}")

        # Visual 3D
        visualize_3d_trajectories(df_full)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()