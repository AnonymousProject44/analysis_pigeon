import pandas as pd
import numpy as np
import cv2
import argparse
import yaml
import os
from scipy.signal import savgol_filter
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation as Rot
from scipy.fft import rfft, rfftfreq
from stereo_vis import BirdKalmanFilter

# Calibration geometry
def load_calibration_data(config_dir="config"):
    def load_yaml(filename):
        path = os.path.join(config_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Configuration file not found: {path}")
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    cfg_l = load_yaml("left.yaml")
    cfg_r = load_yaml("right.yaml")
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
    bird_ids = df['bird_id'].unique()
    for bid in bird_ids:
        mask = df['bird_id'] == bid
        track = df[mask].sort_values('frame')
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
        df.loc[mask, 'y_smooth'] = y_smooth
        df.loc[mask, 'x_smooth'] = x_smooth
        df.loc[mask, 'vy'] = vy
        df.loc[mask, 'vx'] = vx
    return df

def match_stereo_tracks_advanced(df_l, df_r, max_y_error=200.0, min_overlap=15, weight_y=2.0, weight_corr=40.0, max_disparity=500):
    ids_l = df_l['bird_id'].unique()
    ids_r = df_r['bird_id'].unique()
    cost_matrix = np.full((len(ids_l), len(ids_r)), np.inf)
    for i, id_l in enumerate(ids_l):
        track_l = df_l[df_l['bird_id'] == id_l].sort_values('frame')
        for j, id_r in enumerate(ids_r):
            track_r = df_r[df_r['bird_id'] == id_r].sort_values('frame')
            merged = pd.merge(track_l, track_r, on='frame', suffixes=('_L', '_R'))
            if len(merged) < min_overlap: continue
            disparity = merged['x_rect_L'] - merged['x_rect_R']
            mean_disp = disparity.mean()
            if mean_disp < -20 or mean_disp > max_disparity: continue
            y_diff = np.abs(merged['y_rect_L'] - merged['y_rect_R'])
            mean_y_error = y_diff.mean()
            if mean_y_error > max_y_error: continue
            std_l, std_r = merged['vy_L'].std(), merged['vy_R'].std()
            corr_vy = np.corrcoef(merged['vy_L'], merged['vy_R'])[0, 1] if std_l > 1e-4 and std_r > 1e-4 else 0.5
            if np.isnan(corr_vy): corr_vy = 0.0
            if merged['y_smooth_L'].std() > 1e-3 and merged['y_smooth_R'].std() > 1e-3:
                corr_shape = np.corrcoef(merged['y_smooth_L'], merged['y_smooth_R'])[0, 1]
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
    parser.add_argument('--gt_mode', action='store_true')
    parser.add_argument('--clip', type=str, default="006")
    parser.add_argument('--mode', type=str, default='event_frame')
    args = parser.parse_args()

    suffix = "ts" if args.mode == 'time_surface' else "evf"
    left_csv = f"csv/bird_tracking_data_left_{args.clip}_{suffix}.csv"
    right_csv = f"csv/bird_tracking_data_right_{args.clip}_{suffix}.csv"
    output_csv = f"csv/bird_tracking_stereo_data_advanced_{args.clip}_{suffix}.csv"

    try:
        df_l, df_r = pd.read_csv(left_csv), pd.read_csv(right_csv)
        K1, D1, K2, D2, R, T, _, _, _, _ = load_calibration_data()
        (R1, P1), (R2, P2) = get_rectification_matrices(K1, D1, K2, D2, R, T)
        df_l, df_r = batch_rectify_points(df_l, K1, D1, R1, P1), batch_rectify_points(df_r, K2, D2, R2, P2)
        
        if args.gt_mode:
            results = pd.read_csv(f"csv/id_matches_gt_{args.clip}.csv")
        else:
            df_l, df_r = extract_features(df_l), extract_features(df_r)
            results = match_stereo_tracks_advanced(df_l, df_r)
            if not results.empty: results.to_csv(f"csv/id_matches_{args.clip}_{suffix}.csv", index=False)

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
                    ts = row['timestamp_L'] if 'timestamp_L' in row else row['frame'] * 0.005
                    stereo_data.append({'frame': int(row['frame']), 'timestamp': ts, 'bird_id': id_l, 'bird_id_R': id_r, 'x_m': pts_3d[0, k], 'y_m': pts_3d[1, k], 'z_m': pts_3d[2, k]})
            
            df_out = pd.DataFrame(stereo_data)
            unique_birds = df_out['bird_id'].unique()
            for bid in unique_birds:
                mask = df_out['bird_id'] == bid
                indices = df_out[mask].sort_values('timestamp').index
                if len(indices) < 3: continue
                kf = BirdKalmanFilter(dt=0.005)
                for idx in indices:
                    sx, sy, sz = kf.filter_point(df_out.at[idx, 'x_m'], df_out.at[idx, 'y_m'], df_out.at[idx, 'z_m'])
                    df_out.at[idx, 'x_m_smooth'], df_out.at[idx, 'y_m_smooth'], df_out.at[idx, 'z_m_smooth'] = sx, sy, sz

            # STATS CALCULATION
            stats = []
            for bid in unique_birds:
                b_df = df_out[df_out['bird_id'] == bid].sort_values('timestamp')
                vx = np.gradient(b_df['x_m_smooth'], b_df['timestamp'])
                vy = np.gradient(b_df['y_m_smooth'], b_df['timestamp'])
                vz = np.gradient(b_df['z_m_smooth'], b_df['timestamp'])
                b_df['speed'] = np.sqrt(vx**2 + vy**2 + vz**2)
                stats.append({'bird_id': bid, 'avg_speed': b_df['speed'].mean(), 'wingbeat_hz': estimate_wingbeat(b_df)})
            
            df_stats = pd.DataFrame(stats)
            df_out.to_csv(output_csv, index=False)
            df_stats.to_csv(f"csv/bird_flight_stats_{args.clip}.csv", index=False)
            print(f"Global Avg Wingbeat: {df_stats['wingbeat_hz'].median():.2f} Hz")
            print(df_stats)

    except Exception as e:
        print(f"ERROR: {e}")