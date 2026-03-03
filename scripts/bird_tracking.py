import os
import cv2
import csv
import math
import numpy as np
import argparse
import random
from ultralytics import YOLO
from metavision_core.event_io import EventsIterator

# CONFIGURATION
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH_TS = os.path.join(SCRIPT_DIR, "../config/yolo/ts.pt")
MODEL_PATH_EVF = os.path.join(SCRIPT_DIR, "../config/yolo/evf.pt")
CONFIDENCE = 0.5
VALIDATION_COUNT = 3
MAX_GAP_FRAMES = 15
BORDER_MARGIN = 25 
VELOCITY_SMOOTHING = 0.1 
MIN_TOTAL_DURATION = 15
KNOWN_MEDIAN_VELOCITY = 1.73  
VELOCITY_TOLERANCE_PCT = 400  
MAX_ALLOWED_STEP = KNOWN_MEDIAN_VELOCITY * (1 + VELOCITY_TOLERANCE_PCT / 100.0)
HARD_MAX_DIST = 50 

class AssociationTracker:
    def __init__(self, min_hits, width, height):
        self.tracks = []
        self.history = []
        self.min_hits = min_hits
        self.width = width
        self.height = height
        self.id_count = 0
        
    def is_near_border(self, center):
        cx, cy = center
        return (cx < BORDER_MARGIN or cx > self.width - BORDER_MARGIN or 
                cy < BORDER_MARGIN or cy > self.height - BORDER_MARGIN)

    def update(self, detections, frame_idx):
        new_tracks = []
        current_centers = []
        
        for item in detections:
            bbox = item['bbox']
            center = item['center'] 
            
            x1, y1, x2, y2 = bbox
            w, h = x2 - x1, y2 - y1
            
            current_centers.append({'center': center, 'bbox': bbox, 'size': (w, h), 'matched': False})

        for track in self.tracks:
            predicted_center = np.array(track['center']) + track['smooth_vel']
            best_match = None
            min_dist = float('inf')
            
            for i, det in enumerate(current_centers):
                if det['matched']: continue
                dist = np.linalg.norm(predicted_center - np.array(det['center']))
                if dist > MAX_ALLOWED_STEP: continue
                if dist < min_dist:
                    min_dist = dist
                    best_match = i

            if best_match is not None:
                current_centers[best_match]['matched'] = True
                new_center = np.array(current_centers[best_match]['center'])
                
                if self.is_near_border(new_center):
                    self.history.append(track)
                    continue 
                
                inst_vel = new_center - np.array(track['center'])
                track['smooth_vel'] = (VELOCITY_SMOOTHING * inst_vel) + ((1 - VELOCITY_SMOOTHING) * track['smooth_vel'])
                track['center'] = tuple(new_center)
                track['bbox'] = current_centers[best_match]['bbox']
                track['path'].append((tuple(new_center), frame_idx, current_centers[best_match]['size']))
                track['hits'] += 1
                track['age'] = 0
                new_tracks.append(track)
            else:
                track['age'] += 1
                track['center'] = tuple(np.array(track['center']) + track['smooth_vel'])
                if track['age'] <= 3: 
                    new_tracks.append(track)
                else:
                    self.history.append(track)

        for det in current_centers:
            if not det['matched'] and not self.is_near_border(det['center']):
                self.id_count += 1
                bird_color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
                new_tracks.append({
                    'id': self.id_count, 'center': det['center'], 'bbox': det['bbox'],
                    'smooth_vel': np.array([0.0, 0.0]), 
                    'path': [(det['center'], frame_idx, det['size'])], 
                    'hits': 1, 'age': 0, 'color': bird_color
                })
        self.tracks = new_tracks
        return [t for t in self.tracks if t['hits'] >= self.min_hits]

    def bridge_trajectories(self):
        print("Starting trajectory bridging process...")
        all_t = self.history + self.tracks
        all_t = [t for t in all_t if len(t['path']) >= self.min_hits]
        all_t.sort(key=lambda x: x['path'][0][1])
        merged = []
        used_indices = set()
        
        for i in range(len(all_t)):
            if i in used_indices: continue
            curr = all_t[i]
            
            if self.is_near_border(curr['path'][-1][0]):
                merged.append(curr)
                continue

            found_connection = True
            while found_connection:
                found_connection = False
                if self.is_near_border(curr['path'][-1][0]): break

                for j in range(i + 1, len(all_t)):
                    if j in used_indices: continue
                    nxt = all_t[j]
                    
                    gap = nxt['path'][0][1] - curr['path'][-1][1]
                    if gap <= 0 or gap > MAX_GAP_FRAMES: continue 
                    
                    last_pos = np.array(curr['path'][-1][0])
                    next_start_pos = np.array(nxt['path'][0][0])
                    
                    raw_dist = np.linalg.norm(next_start_pos - last_pos)
                    if raw_dist > HARD_MAX_DIST: continue 

                    displacement_vector = next_start_pos - last_pos
                    velocity_vector = curr['smooth_vel']
                    vel_norm = np.linalg.norm(velocity_vector)
                    
                    if vel_norm > 0.5:
                        dot_product = np.dot(displacement_vector, velocity_vector)
                        if dot_product < 0: continue

                    est_pos = last_pos + (curr['smooth_vel'] * gap)
                    proj_dist = np.linalg.norm(est_pos - next_start_pos)
                    
                    if proj_dist < (MAX_ALLOWED_STEP * gap): 
                        curr['path'].extend(nxt['path'])
                        curr['smooth_vel'] = nxt['smooth_vel'] 
                        used_indices.add(j)
                        found_connection = True
                        break 
            merged.append(curr)
        
        final_list = [t for t in merged if len(t['path']) >= MIN_TOTAL_DURATION]
        return final_list

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('raw_file', type=str, help="Choose .raw clip to process")
    parser.add_argument('--mode', type=str, default='event_frame', help="Mode of processing: 'time_surface' or 'event_frame'")
    parser.add_argument('--camera', type=str, default='left', help="Processing camera: 'left' or 'right'")
    parser.add_argument('--dt', type=int, default=5000, help="Delta time in microseconds")
    parser.add_argument('--save_csv', type=str, default='true')

    args = parser.parse_args()
    save_csv_bool = args.save_csv.lower() == 'true'

    if args.mode not in ['time_surface', 'event_frame']:
        print("Invalid mode choice. Use 'time_surface' or 'event_frame'.")
        return
    if args.camera not in ['left', 'right']:
        print("Invalid camera choice. Use 'left' or 'right'.")
        return

    event_file_path = args.raw_file
    raw_name = os.path.splitext(os.path.basename(event_file_path))[0]
    dt = args.dt
    
    if args.mode == 'time_surface':
        filetype = "ts"
        model = YOLO(MODEL_PATH_TS)
    elif args.mode == 'event_frame':
        filetype = "evf"
        model = YOLO(MODEL_PATH_EVF)

    csv_dir = os.path.join(SCRIPT_DIR, "../csv")
    os.makedirs(csv_dir, exist_ok=True)
    filename = os.path.join(csv_dir, f"tracking_{filetype}_{raw_name}_{args.camera}.csv")

    mv_it = EventsIterator(input_path=event_file_path, delta_t=dt)
    height, width = mv_it.get_size()
    
    tracker = AssociationTracker(min_hits=VALIDATION_COUNT, width=width, height=height)
    if args.mode == 'time_surface':
        ts_surface = np.zeros((height, width), dtype=np.uint64)

    frame_idx = 0

    print("Processing...")
    for evs in mv_it:
        if len(evs['x']) == 0: continue
        if args.mode == 'time_surface':
            x, y, t = evs['x'], evs['y'], evs['t']
            ts_surface[y, x] = t
            intensity = 255 * (1.0 - (np.clip(t[-1] - ts_surface, 0, dt) / dt))
            display_bgr = cv2.cvtColor(intensity.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        elif args.mode == 'event_frame':

            event_frame = np.full((height, width, 3), 255, dtype=np.uint8)

            x = evs['x']
            y = evs['y']
            p = evs['p']

            pos_mask = (p == 1)
            neg_mask = (p == 0)

            event_frame[y[pos_mask], x[pos_mask]] = [255, 0, 0]
            event_frame[y[neg_mask], x[neg_mask]] = [0, 0, 255]

            display_bgr = event_frame.copy()
        
        results = model.predict(display_bgr, imgsz=1024, conf=CONFIDENCE, verbose=False)
        detection_list = []
        
        for result in results:
            if result.boxes is None: continue
            
            boxes = result.boxes.xyxy.cpu().numpy()
            
            if result.masks is not None:
                
                for box in boxes:
                    
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
                        'size': (w, h)
                    })
            
            else:
                for box in boxes:
                    cx = (box[0] + box[2]) / 2
                    cy = (box[1] + box[3]) / 2
                    detection_list.append({'bbox': box, 'center': (cx, cy)})
        
        tracker.update(detection_list, frame_idx)
        
        frame_idx += 1

    final_trajs = tracker.bridge_trajectories()
    print(f"Post-processing complete. Found {len(final_trajs)} valid bird paths.")

    if save_csv_bool:
        save_trajectories_to_csv(final_trajs, filename)

if __name__ == "__main__":
    main()