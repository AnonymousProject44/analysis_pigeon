import pygame
import sys
import cv2
import numpy as np
import yaml
import math
import os
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Configuration
WIDTH, HEIGHT = 1000, 800
FPS = 30
YAML_FILE = "trajectories.yaml"
MODEL_PATH = "/home/luisgs44/pigeon_detection/PigeonColor.v1i.yolov11/runs/segment/runs/train/birds_test_run/weights/best.pt"
CONFIDENCE = 0.45 
METRIC_THRESHOLD = 50.0 

# Speed controls
VISUALIZATION = True   
SAVE_VIDEO = False   

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NANO_BACKBONE_PATH = os.path.join(SCRIPT_DIR, "nano_tracker", "nanotrack_backbone_sim.onnx")
NANO_HEAD_PATH = os.path.join(SCRIPT_DIR, "nano_tracker", "nanotrack_head_sim.onnx")

# Strict bounds
def is_strictly_in_bounds(bbox):
    if bbox is None: return False
    x1, y1, x2, y2 = map(int, bbox)
    if x1 <= 0 or y1 <= 0 or x2 >= WIDTH or y2 >= HEIGHT:
        return False
    return True

# Custom tracker
class CustomTracker:
    def __init__(self):
        self.tracks = []
        self.id_count = 0
        self.min_hits = 3

    def update(self, detections, _):
        current_centers = [{'center': d['center'], 'bbox': d['bbox'], 'matched': False} for d in detections]
        new_tracks = []
        
        for track in self.tracks:
            pred = np.array(track['center']) + track['vel']
            best_match, min_dist = None, float('inf')
            
            for i, det in enumerate(current_centers):
                if det['matched']: continue
                dist = np.linalg.norm(pred - np.array(det['center']))
                if dist < 50 and dist < min_dist:
                    min_dist = dist
                    best_match = i
            
            if best_match is not None:
                current_centers[best_match]['matched'] = True
                new_c = np.array(current_centers[best_match]['center'])
                inst_vel = new_c - np.array(track['center'])
                track['vel'] = (0.1 * inst_vel) + (0.9 * track['vel'])
                track['center'] = tuple(new_c)
                track['bbox'] = current_centers[best_match]['bbox']
                track['hits'] += 1
                new_tracks.append(track)
            else:
                track['center'] = tuple(np.array(track['center']) + track['vel'])
                x1,y1,x2,y2 = track['bbox']
                vx,vy = track['vel']
                track['bbox'] = (x1+vx, y1+vy, x2+vx, y2+vy)
                new_tracks.append(track) 

        for det in current_centers:
            if not det['matched']:
                self.id_count += 1
                new_tracks.append({
                    'id': self.id_count, 'center': det['center'], 'bbox': det['bbox'],
                    'vel': np.array([0.,0.]), 'hits': 1
                })
        
        self.tracks = [t for t in new_tracks if is_strictly_in_bounds(t['bbox'])]
        return [t for t in self.tracks if t['hits'] >= self.min_hits]

# OpenCV Manager
class OpenCVBirdManager:
    def __init__(self, algorithm_name):
        self.trackers = []
        self.algo = algorithm_name
        self.nano_params = None
        self.active_status = [] 

        if self.algo == "NANO":
            if os.path.exists(NANO_BACKBONE_PATH) and os.path.exists(NANO_HEAD_PATH):
                self.nano_params = cv2.TrackerNano_Params()
                self.nano_params.backbone = NANO_BACKBONE_PATH
                self.nano_params.neckhead = NANO_HEAD_PATH
            else:
                self.algo = "MIL"

    def create_algo(self):
        if self.algo == "CSRT": return cv2.TrackerCSRT_create()
        if self.algo == "KCF": return cv2.TrackerKCF_create()
        if self.algo == "NANO": return cv2.TrackerNano_create(self.nano_params)
        return cv2.TrackerMIL_create()

    def initialize(self, frame, initial_boxes):
        self.trackers = []
        self.active_status = []
        for box in initial_boxes:
            try:
                tracker = self.create_algo()
                x1, y1, x2, y2 = map(int, box)
                w, h = x2-x1, y2-y1
                tracker.init(frame, (x1, y1, w, h))
                self.trackers.append(tracker)
                self.active_status.append(True)
            except:
                self.active_status.append(False)

    def update(self, frame):
        boxes = []
        for i, tr in enumerate(self.trackers):
            if not self.active_status[i]:
                boxes.append(None)
                continue
            success, box = tr.update(frame)
            if success:
                x, y, w, h = map(int, box)
                bbox = (x, y, x+w, y+h)
                if is_strictly_in_bounds(bbox):
                    boxes.append(box)
                else:
                    self.active_status[i] = False
                    boxes.append(None) 
            else:
                self.active_status[i] = False
                boxes.append(None)
        return boxes

# Bird YAML
class YamlBird(pygame.sprite.Sprite):
    def __init__(self, config_data, bird_id):
        super().__init__()
        self.id = bird_id
        self.sprites = []
        try:
            for i in range(1, 6):
                img = pygame.image.load(f"images_animation/bird{i}.png").convert_alpha()
                self.sprites.append(img)
        except:
            s = pygame.Surface((40,40)); s.fill((0,0,0)); self.sprites = [s]

        start, end, speed = config_data['start'], config_data['end'], config_data['speed']
        self.exact_x, self.exact_y = float(start[0]), float(start[1])
        dx, dy = end[0] - start[0], end[1] - start[1]
        dist = math.sqrt(dx**2 + dy**2)
        self.vx, self.vy = (dx/dist)*speed, (dy/dist)*speed if dist > 0 else (0,0)
        
        if self.vx < 0: self.sprites = [pygame.transform.flip(s,True,False) for s in self.sprites]
        self.image = self.sprites[0]
        self.rect = self.image.get_rect(center=(start[0], start[1]))
        self.frame_idx = 0

    def update(self):
        self.exact_x += self.vx; self.exact_y += self.vy
        self.rect.centerx = int(self.exact_x); self.rect.centery = int(self.exact_y)
        self.frame_idx += 0.25
        self.image = self.sprites[int(self.frame_idx)%len(self.sprites)]
        if (self.rect.right < 0 or self.rect.left > WIDTH or 
            self.rect.bottom < 0 or self.rect.top > HEIGHT):
            self.kill()

# Main
def main():
    # Force disable video if no visualization (saves huge time)
    global SAVE_VIDEO
    if not VISUALIZATION:
        SAVE_VIDEO = False
        print(">> VISUALIZATION IS OFF. Running in FAST MODE (No Video, No Window Update).")
    pygame.init()
    # Window context for textures, even if we don't show it much
    if VISUALIZATION:
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
    else:
        screen = pygame.display.set_mode((WIDTH, HEIGHT), flags=pygame.HIDDEN)

    pygame.display.set_caption("IROS BENCHMARK RUNNER")
    font = pygame.font.SysFont("Arial", 12, bold=True)
    
    print("Loading YOLO...")
    try: model = YOLO(MODEL_PATH)
    except: model = YOLO("yolov8n.pt")

    print(f"Reading scenarios from {YAML_FILE}...")
    try:
        with open(YAML_FILE, 'r') as f: config = yaml.safe_load(f)
        scenarios = config['scenarios']
    except Exception as e:
        print(e); return

    clock = pygame.time.Clock()
    
    video_writer = None
    if SAVE_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        video_writer = cv2.VideoWriter('benchmark_metrics.mp4', fourcc, FPS, (WIDTH, HEIGHT))

    full_metrics = []

    for idx, scenario in enumerate(scenarios):
        scen_name = scenario.get('name', f"Scenario {idx+1}")
        print(f"\n--- Processing: {scen_name} ---")

        ground_truth_birds = pygame.sprite.Group()
        gt_list = []
        initial_bboxes = []
        
        for i, b_conf in enumerate(scenario['birds']):
            bird = YamlBird(b_conf, i)
            ground_truth_birds.add(bird)
            gt_list.append(bird)
            initial_bboxes.append((bird.rect.left, bird.rect.top, bird.rect.right, bird.rect.bottom))

        my_tracker = CustomTracker()
        tracker_csrt = OpenCVBirdManager("CSRT")
        tracker_kcf = OpenCVBirdManager("KCF")
        tracker_nano = OpenCVBirdManager("NANO") 

        running = True
        initialized = False
        frame_cnt = 0

        while running:
            for event in pygame.event.get(): 
                if event.type == pygame.QUIT: running = False; sys.exit()

            ground_truth_birds.update()
            if len(ground_truth_birds) == 0: running = False

            screen.fill((255, 255, 255))
            ground_truth_birds.draw(screen)

            # Capture frame from Pygame Surface
            view = pygame.surfarray.array3d(screen)
            frame_cv = cv2.cvtColor(np.transpose(view, (1, 0, 2)), cv2.COLOR_RGB2BGR)

            # Custom tracker
            results = model.predict(frame_cv, conf=CONFIDENCE, verbose=False)
            detections = []
            for r in results:
                if r.boxes:
                    for box in r.boxes.xyxy.cpu().numpy():
                        x1,y1,x2,y2 = map(int, box)
                        cx, cy = (x1+x2)/2, (y1+y2)/2
                        detections.append({'bbox':(x1,y1,x2,y2), 'center':(cx,cy)})
            my_results_raw = my_tracker.update(detections, 0)
            
            # OpenCV
            if not initialized:
                tracker_csrt.initialize(frame_cv, initial_bboxes)
                tracker_kcf.initialize(frame_cv, initial_bboxes)
                tracker_nano.initialize(frame_cv, initial_bboxes)
                initialized = True
            
            res_csrt = tracker_csrt.update(frame_cv)
            res_kcf = tracker_kcf.update(frame_cv)
            res_nano = tracker_nano.update(frame_cv)

            # Data Collection
            for i, gt_bird in enumerate(gt_list):
                if not gt_bird.alive(): continue
                
                gt_cx, gt_cy = gt_bird.rect.centerx, gt_bird.rect.centery
                
                def get_error(res_bbox, is_yolo=False):
                    if res_bbox is None: return np.nan
                    if is_yolo:
                        bx = res_bbox
                        cx, cy = (bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2
                    else:
                        x, y, w, h = res_bbox
                        cx, cy = x + w/2, y + h/2
                    return math.sqrt((cx-gt_cx)**2 + (cy-gt_cy)**2)

                # Calculate errors for all trackers
                err_custom = min([get_error(t['bbox'], True) for t in my_results_raw] + [float('inf')])
                err_csrt = get_error(res_csrt[i] if i < len(res_csrt) else None)
                err_kcf = get_error(res_kcf[i] if i < len(res_kcf) else None)
                err_nano = get_error(res_nano[i] if i < len(res_nano) else None)

                # Tracker data map
                tracker_data = [
                    ('MOE (Ours)', err_custom),
                    ('CSRT', err_csrt),
                    ('KCF', err_kcf),
                    ('NANO', err_nano)
                ]

                for name, err in tracker_data:
                    is_valid = not np.isnan(err) and err < METRIC_THRESHOLD
                    full_metrics.append({
                        'Scenario': scen_name,
                        'traj': scen_name, 
                        'Tracker': name, 
                        'Error': err if is_valid else np.nan, 
                        'Success': is_valid
                    })


            if VISUALIZATION:
                for t in my_results_raw:
                    bx = t['bbox']
                    pygame.draw.rect(screen, (0, 255, 0), (bx[0], bx[1], bx[2]-bx[0], bx[3]-bx[1]), 3)
                    screen.blit(font.render(f"ME:{t['id']}", True, (0,150,0)), (bx[0], bx[1]-15))
                
                def draw_box(res, color, label, offset):
                    for _, box in enumerate(res):
                        if box:
                            x, y, w, h = map(int, box)
                            pygame.draw.rect(screen, color, (x-offset, y-offset, w+(offset*2), h+(offset*2)), 2)
                            screen.blit(font.render(label, True, color), (x, y + h + offset + 2))

                draw_box(res_csrt, (0,0,255), "CSRT", 2)
                draw_box(res_kcf, (255,0,0), "KCF", -4)
                draw_box(res_nano, (255,140,0), "NANO", 6)
                
                pygame.draw.rect(screen, (240,240,240), (0,0,240,110))
                screen.blit(font.render(f"SCENARIO: {scen_name}", True, (0,0,0)), (10, 10))
                screen.blit(font.render("GREEN: MOE", True, (0,150,0)), (10, 30))
                screen.blit(font.render("BLUE: CSRT", True, (0,0,255)), (10, 50))
                screen.blit(font.render("RED: KCF", True, (255,0,0)), (10, 70))
                screen.blit(font.render("ORANGE: NANO", True, (200,100,0)), (10, 90))

                pygame.display.flip()
                if video_writer:
                    video_writer.write(frame_cv) # Save raw clean frame or screen? Usually screen with boxes
                    # Re-capture screen if we want boxes
                    view_final = pygame.surfarray.array3d(screen)
                    video_writer.write(cv2.cvtColor(np.transpose(view_final, (1, 0, 2)), cv2.COLOR_RGB2BGR))
                
                clock.tick(FPS)
            else:
                # FAST MODE: Just print progress every 30 frames
                if frame_cnt % 30 == 0:
                    print(f"  -> Frame {frame_cnt} processed...", end='\r')

            frame_cnt += 1

        print(f"  -> Done. Total frames: {frame_cnt}")

    if video_writer: video_writer.release()
    pygame.quit()

    # Results
    print("Calculating metrics and saving CSV...")
    df = pd.DataFrame(full_metrics)

    # Save to CSV with the new 'traj' column included
    df.to_csv('benchmark_results.csv', index=False)
    print("File 'benchmark_results.csv' saved successfully.")
    
    # Overall
    summary = df.groupby('Tracker').agg(
        Precision_px=('Error', 'mean'),
        Robustness_pct=('Success', 'mean')
    ).reset_index()
    summary['Robustness_pct'] *= 100
    
    print("\nOverall performance:")
    print(summary.to_string(index=False, float_format="%.2f"))

    # Per Scenario
    pivot_table = df.pivot_table(index='Scenario', columns='Tracker', values='Success', aggfunc='mean') * 100
    print("\nRobustness per scenario (%):")
    print(pivot_table.to_string(float_format="%.1f"))

    try:
        # Set the backend to PGF
        matplotlib.use("pgf")
        matplotlib.rcParams.update({
            "pgf.texsystem": "pdflatex",
            "font.family": "serif",
            "text.usetex": True,
            "pgf.rcfonts": False,
        })

        sns.set_theme(style="whitegrid")
        
        # Precision Plot 
        plt.figure(figsize=(5, 3)) 
        sns.barplot(data=df, x='Tracker', y='Error', palette="viridis", errorbar=None)
        plt.ylabel('Mean Error (px)')
        plt.xlabel('Tracking Algorithm')
        plt.tight_layout()
        plt.savefig('benchmark_precision.pgf') 

        # Robustness Plot 
        plt.figure(figsize=(16, 9)) # Increased height slightly to accommodate legend below

        scen_sum = df.groupby(['Scenario', 'Tracker'])['Success'].mean().reset_index()
        scen_sum['Success'] *= 100
        
        # Create the barplot
        ax = sns.barplot(data=scen_sum, x='Scenario', y='Success', hue='Tracker')
        
        ax.set_ylabel('')
        ax.set_xlabel('')
        
        ax.tick_params(axis='both', which='major', labelsize=28)
        # Determine number of trackers for horizontal layout
        num_trackers = scen_sum['Tracker'].nunique()
        
        # Position legend below the plot (loc='upper center', bbox_to_anchor=(0.5, -0.2))
        plt.legend(
            title='', 
            loc='upper center', 
            bbox_to_anchor=(0.5, -0.1), 
            ncol=num_trackers, 
            frameon=True,
            fontsize=24
        )
        
        plt.tight_layout()
        
        # Save the figure
        plt.savefig('benchmark_robustness.pgf', bbox_inches='tight')
        plt.savefig('benchmark_robustness.png', dpi=300, bbox_inches='tight')
    except Exception as e:
        print(f"Failed to save PGF plots: {e}")

if __name__ == "__main__":
    main()