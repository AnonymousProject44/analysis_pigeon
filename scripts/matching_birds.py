import pandas as pd
import numpy as np
import cv2
import argparse
import yaml
import sys
import os
from scipy.signal import savgol_filter
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation as Rot
from scipy.fft import rfft, rfftfreq
from stereo_visualizer import BirdKalmanFilter, filter_outliers
from naming import tag_from_tracking_csv, session_from_tag

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Calibration geometry
def load_calibration_data(config_dir="../config/calibration/", extrinsics_path=None):
    def load_yaml(filename):
        path = os.path.join(config_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Configuration file not found: {path}")
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    cfg_l = load_yaml("left.yaml")
    cfg_r = load_yaml("right.yaml")
    if extrinsics_path:
        with open(extrinsics_path, 'r') as f:
            cfg_sys = yaml.safe_load(f)
    else:
        cfg_sys = load_yaml("extrinsics.yaml")

    K1 = np.array(cfg_l['camera_matrix']['data'], dtype=np.float32).reshape(3, 3)
    D1 = np.array(cfg_l['distortion_coefficients']['data'], dtype=np.float32)
    K2 = np.array(cfg_r['camera_matrix']['data'], dtype=np.float32).reshape(3, 3)
    D2 = np.array(cfg_r['distortion_coefficients']['data'], dtype=np.float32)
    
    W = cfg_sys['image_width']
    H = cfg_sys['image_height']
    rot_order = cfg_sys['rotation_order']
    
    yaw_L, pitch_L, roll_L = cfg_sys['yaw_L'], cfg_sys['pitch_L'], cfg_sys['roll_L']
    yaw_R, pitch_R, roll_R = cfg_sys['yaw_R'], cfg_sys['pitch_R'], cfg_sys['roll_R']
    
    r_l = Rot.from_euler(rot_order, [yaw_L, pitch_L, roll_L], degrees=True).as_matrix()
    r_r = Rot.from_euler(rot_order, [yaw_R, pitch_R, roll_R], degrees=True).as_matrix()

    baseline = cfg_sys['baseline']
    height_diff = cfg_sys['height_diff']
    pos_right_rel_left = np.array([baseline, height_diff, 0.0]) 

    R = r_r @ r_l.T
    T = r_r @ (-pos_right_rel_left).reshape(3, 1)

    return K1, D1, K2, D2, R.astype(np.float64), T.astype(np.float64), (W, H), r_l, r_r, pos_right_rel_left

def get_rectification_matrices(K1, D1, K2, D2, R, T, image_size=(1280, 720)):
    R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(
        K1, D1, K2, D2, image_size, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
    )
    return (R1, P1), (R2, P2)

def batch_rectify_points(df, K, D, R_rect, P_rect):
    if df.empty: return df
    centers_x = df['x'].values + df['w'].values / 2.0
    centers_y = df['y'].values + df['h'].values / 2.0
    pts_raw = np.column_stack((centers_x, centers_y)).astype(np.float32).reshape(-1, 1, 2)
    rect_pts = cv2.undistortPoints(pts_raw, K, D, R=R_rect, P=P_rect)
    rect_pts = rect_pts.reshape(-1, 2)
    df['x_rect'] = rect_pts[:, 0]
    df['y_rect'] = rect_pts[:, 1]
    return df

def extract_features(df, window_length=21, polyorder=2):
    df = df.copy()
    for col in ['y_smooth', 'vy', 'x_smooth', 'vx']:
        df[col] = 0.0
    for bid, group in df.groupby('bird_id'):
        track = group.sort_values('frame')
        if len(track) < 5: continue
        y_raw = track['y_rect'].values
        x_raw = track['x_rect'].values
        wl = min(window_length, len(track))
        if wl % 2 == 0: wl -= 1
        if wl < 5: wl = 3
        y_smooth = savgol_filter(y_raw, wl, polyorder)
        x_smooth = savgol_filter(x_raw, wl, polyorder)
        vy = np.gradient(y_smooth)
        vx = np.gradient(x_smooth)
        df.loc[track.index, 'y_smooth'] = y_smooth
        df.loc[track.index, 'x_smooth'] = x_smooth
        df.loc[track.index, 'vy'] = vy
        df.loc[track.index, 'vx'] = vx
    return df

def _tracks_as_arrays(df):
    """Per-bird frame-sorted numpy arrays of the columns the matcher needs."""
    tracks = {}
    for bid, group in df.groupby('bird_id'):
        g = group.sort_values('frame')
        tracks[bid] = {
            'frames': g['frame'].to_numpy(),
            'x_rect': g['x_rect'].to_numpy(),
            'y_rect': g['y_rect'].to_numpy(),
            'vy': g['vy'].to_numpy(),
            'y_smooth': g['y_smooth'].to_numpy(),
        }
    return tracks

def match_stereo_tracks_advanced(df_l, df_r, max_y_error=200.0, min_overlap=15, weight_y=2.0, weight_corr=40.0, max_disparity=500):
    ids_l = df_l['bird_id'].unique()
    ids_r = df_r['bird_id'].unique()
    tracks_l = _tracks_as_arrays(df_l)
    tracks_r = _tracks_as_arrays(df_r)
    cost_matrix = np.full((len(ids_l), len(ids_r)), np.inf)
    for i, id_l in enumerate(ids_l):
        tl = tracks_l[id_l]
        for j, id_r in enumerate(ids_r):
            tr = tracks_r[id_r]
            # Cheap reject: the frame ranges cannot overlap enough
            if min(tl['frames'][-1], tr['frames'][-1]) - max(tl['frames'][0], tr['frames'][0]) + 1 < min_overlap:
                continue
            _, ia, ib = np.intersect1d(tl['frames'], tr['frames'], return_indices=True)
            if len(ia) < min_overlap: continue
            disparity = tl['x_rect'][ia] - tr['x_rect'][ib]
            mean_disp = disparity.mean()
            if mean_disp < -20 or mean_disp > max_disparity: continue
            y_diff = np.abs(tl['y_rect'][ia] - tr['y_rect'][ib])
            mean_y_error = y_diff.mean()
            if mean_y_error > max_y_error: continue
            vy_l, vy_r = tl['vy'][ia], tr['vy'][ib]
            std_l, std_r = np.std(vy_l, ddof=1), np.std(vy_r, ddof=1)
            corr_vy = np.corrcoef(vy_l, vy_r)[0, 1] if std_l > 1e-4 and std_r > 1e-4 else 0.5
            if np.isnan(corr_vy): corr_vy = 0.0
            ys_l, ys_r = tl['y_smooth'][ia], tr['y_smooth'][ib]
            if np.std(ys_l, ddof=1) > 1e-3 and np.std(ys_r, ddof=1) > 1e-3:
                corr_shape = np.corrcoef(ys_l, ys_r)[0, 1]
            else: corr_shape = 0.5
            total_cost = (mean_y_error / max_y_error * weight_y) + ((1.0 - corr_vy) * weight_corr) + ((1.0 - corr_shape) * (weight_corr * 0.5))
            cost_matrix[i, j] = total_cost
    solver_matrix = np.where(cost_matrix == np.inf, 1e9, cost_matrix)
    row_ind, col_ind = linear_sum_assignment(solver_matrix)
    matches = []
    for row, col in zip(row_ind, col_ind):
        if cost_matrix[row, col] < 1000.0: 
            matches.append({'Left_ID': ids_l[row], 'Right_ID': ids_r[col], 'Total_Cost': round(cost_matrix[row, col], 4)})
    return pd.DataFrame(matches)

def save_trajectory_plot_3d(df_out, colors_by_id, filename):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['CMU Serif', 'Computer Modern Roman', 'DejaVu Serif'],
        'mathtext.fontset': 'cm',
        'font.size': 12,
    })

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Equal axis ranges from the (already outlier-filtered) raw triangulation
    all_x, all_z, all_y = df_out['x_m'], df_out['z_m'], -df_out['y_m']
    mid_x, mid_z, mid_y = (all_x.max()+all_x.min())*0.5, (all_z.max()+all_z.min())*0.5, (all_y.max()+all_y.min())*0.5
    max_range = np.array([all_x.max()-all_x.min(), all_z.max()-all_z.min(), all_y.max()-all_y.min()]).max()/2.0
    lim_x = (mid_x-max_range, mid_x+max_range)
    lim_z = (mid_z-max_range, mid_z+max_range)
    lim_y = (mid_y-max_range, mid_y+max_range)

    handles = []
    for bid in sorted(df_out['bird_id'].unique()):
        b = df_out[df_out['bird_id'] == bid].sort_values('timestamp')
        if len(b) < 2: continue
        # Prefer the Kalman-smoothed positions, fall back to raw triangulation
        xs = b['x_m_smooth'].combine_first(b['x_m']).values if 'x_m_smooth' in b else b['x_m'].values
        ys = b['y_m_smooth'].combine_first(b['y_m']).values if 'y_m_smooth' in b else b['y_m'].values
        zs = b['z_m_smooth'].combine_first(b['z_m']).values if 'z_m_smooth' in b else b['z_m'].values
        ys = -ys
        # Blank out points that escape the axis box (diverging Kalman transients) so lines break there
        out = ((xs < lim_x[0]) | (xs > lim_x[1]) | (zs < lim_z[0]) | (zs > lim_z[1]) |
               (ys < lim_y[0]) | (ys > lim_y[1]))
        if out.all(): continue
        xs, ys, zs = xs.copy(), ys.copy(), zs.copy()
        xs[out] = np.nan; ys[out] = np.nan; zs[out] = np.nan
        color = colors_by_id.get(int(bid), (0.2, 0.2, 0.2))
        ax.plot(xs, zs, ys, color=color, linewidth=2.5, solid_capstyle='round')
        handles.append(Line2D([0], [0], color=color, linewidth=8, label=f"Bird {int(bid)}"))

    ax.set_xlabel('X (Lateral) [m]', labelpad=8)
    ax.set_ylabel('Z (Depth) [m]', labelpad=8)
    ax.set_zlabel('Y (Height) [m]', labelpad=8)
    ax.set_xlim(*lim_x)
    ax.set_ylim(*lim_z)
    ax.set_zlim(*lim_y)

    if handles and len(handles) <= 15:
        ax.legend(handles=handles, loc='upper right', framealpha=1.0,
                  edgecolor='black', fancybox=True, handlelength=1.5)
    elif handles:
        ax.set_title(f"{len(handles)} birds", fontsize=12)
    fig.savefig(filename, bbox_inches='tight', pad_inches=0.4)
    plt.close(fig)
    print(f"--- 3D trajectory plot saved to {filename} ---")

def estimate_wingbeat(df_bird, fs=200.0):
    if len(df_bird) < 40: return np.nan
    signal = df_bird['y_m_smooth'].values
    # Detrend to isolate oscillation
    signal_detrended = signal - np.polyval(np.polyfit(np.arange(len(signal)), signal, 1), np.arange(len(signal)))
    n = len(signal_detrended)
    yf = np.abs(rfft(signal_detrended))
    xf = rfftfreq(n, 1 / fs)
    mask = (xf >= 3.0) & (xf <= 18.0)
    if not np.any(mask): return np.nan
    return xf[mask][np.argmax(yf[mask])]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('left_csv', type=str, default="../csv/tracking_ev_clip_006_left.csv", help="Path to the tracking CSV for the left camera")
    parser.add_argument('right_csv', type=str, default="../csv/tracking_ev_clip_006_right.csv", help="Path to the tracking CSV for the right camera")
    parser.add_argument('--clip', type=str, default="006", help="Clip ID")
    parser.add_argument('--mode', type=str, default='event_frame')
    parser.add_argument('--save_plot', action='store_true', help="Save a 3D PDF plot of the matched trajectories")
    parser.add_argument('--dt', type=int, default=10000, help="Frame delta time in microseconds (must match the tracking dt)")
    parser.add_argument('--extrinsics', type=str, default=None, help="Extrinsics YAML (default: auto from ~/Events/Birds/{date}/{spot}/extrinsics.yaml, else config)")
    parser.add_argument('--birds_dir', type=str, default=os.path.expanduser("~/Events/Birds"), help="Dataset root used to auto-locate the session extrinsics")
    args = parser.parse_args()

    if args.mode not in ['time_surface', 'event_frame']:
        print("Invalid mode choice. Use 'time_surface' or 'event_frame'.")
        sys.exit(1)
    
    suffix = "ts" if args.mode == 'time_surface' else "evf"
    tag = tag_from_tracking_csv(args.left_csv) or f"clip_{args.clip}"
    output_csv = os.path.join(SCRIPT_DIR, f"../csv/matching_{tag}_{suffix}.csv")
    config_dir = os.path.join(SCRIPT_DIR, "../config/calibration/")
    dt_s = args.dt * 1e-6

    # Session extrinsics: explicit arg > Events/Birds/{date}/{spot}/extrinsics.yaml > config fallback
    extrinsics_path = args.extrinsics
    if extrinsics_path is None:
        date, spot = session_from_tag(tag)
        if date and spot:
            candidate = os.path.join(args.birds_dir, date, spot, "extrinsics.yaml")
            if os.path.exists(candidate):
                extrinsics_path = candidate
    print(f"Using extrinsics: {extrinsics_path or os.path.join(config_dir, 'extrinsics.yaml')}")

    try:
        df_l, df_r = pd.read_csv(args.left_csv), pd.read_csv(args.right_csv)
        K1, D1, K2, D2, R, T, _, _, _, _ = load_calibration_data(config_dir, extrinsics_path)
        (R1, P1), (R2, P2) = get_rectification_matrices(K1, D1, K2, D2, R, T)
        df_l, df_r = batch_rectify_points(df_l, K1, D1, R1, P1), batch_rectify_points(df_r, K2, D2, R2, P2)
        
        df_l, df_r = extract_features(df_l), extract_features(df_r)
        results = match_stereo_tracks_advanced(df_l, df_r)
        matches_csv = os.path.join(SCRIPT_DIR, f"../csv/id_matches_{tag}_{suffix}.csv")
        if not results.empty: results.to_csv(matches_csv, index=False)

        if not results.empty:
            stereo_data = []
            for _, match in results.iterrows():
                id_l, id_r = int(match['Left_ID']), int(match['Right_ID'])
                t_l, t_r = df_l[df_l['bird_id'] == id_l], df_r[df_r['bird_id'] == id_r]
                merged = pd.merge(t_l, t_r, on='frame', suffixes=('_L', '_R'))
                if merged.empty: continue
                pts_l = merged[['x_rect_L', 'y_rect_L']].to_numpy(dtype=np.float32).T
                pts_r = merged[['x_rect_R', 'y_rect_R']].to_numpy(dtype=np.float32).T
                pts_4d = cv2.triangulatePoints(P1, P2, pts_l, pts_r)
                pts_3d = pts_4d[:3, :] / pts_4d[3, :]
                for k in range(len(merged)):
                    row = merged.iloc[k]
                    ts = row['timestamp_L'] if 'timestamp_L' in row else row['frame'] * dt_s
                    stereo_data.append({'frame': int(row['frame']), 'timestamp': ts, 'bird_id': id_l, 'bird_id_R': id_r, 'x_m': pts_3d[0, k], 'y_m': pts_3d[1, k], 'z_m': pts_3d[2, k]})
            
            df_out = pd.DataFrame(stereo_data)
            unique_birds = df_out['bird_id'].unique()
            for bid in unique_birds:
                mask = df_out['bird_id'] == bid
                indices = df_out[mask].sort_values('timestamp').index
                if len(indices) < 3: continue
                kf = BirdKalmanFilter(dt=dt_s)
                for idx in indices:
                    sx, sy, sz = kf.filter_point(df_out.at[idx, 'x_m'], df_out.at[idx, 'y_m'], df_out.at[idx, 'z_m'])
                    df_out.at[idx, 'x_m_smooth'], df_out.at[idx, 'y_m_smooth'], df_out.at[idx, 'z_m_smooth'] = sx, sy, sz

            # Stats calculation
            stats = []
            for bid in unique_birds:
                b_df = df_out[df_out['bird_id'] == bid].sort_values('timestamp')
                vx = np.gradient(b_df['x_m_smooth'], b_df['timestamp'])
                vy = np.gradient(b_df['y_m_smooth'], b_df['timestamp'])
                vz = np.gradient(b_df['z_m_smooth'], b_df['timestamp'])
                b_df['speed'] = np.sqrt(vx**2 + vy**2 + vz**2)
                stats.append({'bird_id': bid, 'avg_speed': b_df['speed'].mean(), 'wingbeat_hz': estimate_wingbeat(b_df, fs=1.0 / dt_s)})
            
            df_stats = pd.DataFrame(stats)
            df_out.to_csv(output_csv, index=False)

            if args.save_plot:
                # Bird colors from the left tracking CSV so 2D/3D plots and videos match
                colors_by_id = {}
                if {'color_r', 'color_g', 'color_b'}.issubset(df_l.columns):
                    for bid, grp in df_l.groupby('bird_id'):
                        row = grp.iloc[0]
                        colors_by_id[int(bid)] = (row['color_r'] / 255.0, row['color_g'] / 255.0, row['color_b'] / 255.0)
                plot_dir = os.path.join(SCRIPT_DIR, "../plots")
                os.makedirs(plot_dir, exist_ok=True)
                plot_path = os.path.join(plot_dir, f"trajectories3d_{tag}_{suffix}.pdf")
                df_plot = filter_outliers(df_out)
                if df_plot.empty:
                    print("All points filtered as outliers, skipping 3D plot.")
                else:
                    save_trajectory_plot_3d(df_plot, colors_by_id, plot_path)
            stats_csv = os.path.join(SCRIPT_DIR, f"../csv/bird_stats_{tag}_{suffix}.csv")
            df_stats.to_csv(stats_csv, index=False)
            print(f"Global Avg Wingbeat: {df_stats['wingbeat_hz'].median():.2f} Hz")
            print(df_stats)

    except Exception as e:
        print(f"ERROR: {e}")