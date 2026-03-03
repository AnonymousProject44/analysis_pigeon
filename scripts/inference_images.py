import os
import cv2
import numpy as np
import argparse
from ultralytics import YOLO
from metavision_core.event_io import EventsIterator

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH_TS = os.path.join(SCRIPT_DIR, "../config/yolo/ts.pt")
MODEL_PATH_EVF = os.path.join(SCRIPT_DIR, "../config/yolo/evf.pt")
CONFIDENCE = 0.5
DELTA_T = 5000 
DOT_RADIUS = 3
DOT_COLOR = (75, 255, 75) 

def get_centroid(roi, mode):
    # Convert ROI to grayscale for processing
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold based on the specific representation logic
    if mode == 'ts':
        _, mask = cv2.threshold(gray_roi, 30, 255, cv2.THRESH_BINARY)
    else:
        # Event frame has white background (255), so we invert
        _, mask = cv2.threshold(gray_roi, 250, 255, cv2.THRESH_BINARY_INV)
    
    # Calculate moments to find the center of mass
    M = cv2.moments(mask)
    if M['m00'] > 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        return cx, cy
    return None

def process_snapshot(target_frame, event_file):
    # Load models
    model_ts = YOLO(MODEL_PATH_TS)
    model_evf = YOLO(MODEL_PATH_EVF)

    # Initialize iterator
    mv_it = EventsIterator(input_path=event_file, delta_t=DELTA_T)
    height, width = mv_it.get_size()
    ts_surface = np.zeros((height, width), dtype=np.uint64)

    frame_idx = 0
    print(f"Seeking frame {target_frame}...")

    for evs in mv_it:
        if len(evs['x']) == 0:
            continue
        
        # We must update the Time Surface even while skipping to keep it accurate
        x, y, t = evs['x'], evs['y'], evs['t']
        ts_surface[y, x] = t

        if frame_idx < target_frame:
            frame_idx += 1
            continue

        # If we reach here, we are at the target frame
        print(f"Processing target frame: {frame_idx}")

        # 1. Generate Time Surface Representation
        intensity = 255 * (1.0 - (np.clip(t[-1] - ts_surface, 0, DELTA_T) / DELTA_T))
        img_ts = cv2.cvtColor(intensity.astype(np.uint8), cv2.COLOR_GRAY2BGR)

        # 2. Generate Event Frame Representation (White background)
        img_evf = np.full((height, width, 3), 255, dtype=np.uint8)
        p = evs['p']
        img_evf[y[p == 1], x[p == 1]] = [255, 0, 0] # Blue for Positive
        img_evf[y[p == 0], x[p == 0]] = [0, 0, 255] # Red for Negative

        # Inference for Time Surface
        results_ts = model_ts.predict(img_ts, imgsz=1024, conf=CONFIDENCE, verbose=False)
        for res in results_ts:
            if res.boxes is not None:
                for box in res.boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = map(int, box)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width, x2), min(height, y2)
                    roi = img_ts[y1:y2, x1:x2]
                    if roi.size > 0:
                        center = get_centroid(roi, 'ts')
                        if center:
                            cv2.circle(img_ts, (x1 + center[0], y1 + center[1]), DOT_RADIUS, DOT_COLOR, -1)

        # Inference for Event Frame
        results_evf = model_evf.predict(img_evf, imgsz=1024, conf=CONFIDENCE, verbose=False)
        for res in results_evf:
            if res.boxes is not None:
                for box in res.boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = map(int, box)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width, x2), min(height, y2)
                    roi = img_evf[y1:y2, x1:x2]
                    if roi.size > 0:
                        center = get_centroid(roi, 'evf')
                        if center:
                            cv2.circle(img_evf, (x1 + center[0], y1 + center[1]), DOT_RADIUS, DOT_COLOR, -1)

        # Save results
        cv2.imwrite(f"fig/ts_frame_{frame_idx}.png", img_ts)
        cv2.imwrite(f"fig/evf_frame_{frame_idx}.png", img_evf)
        
        # Display side-by-side
        combined = np.hstack((img_ts, img_evf))
        cv2.imshow(f"Frame {frame_idx}: TS vs EVF", combined)
        cv2.waitKey(0)
        
        # Exit after processing the target
        break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('event_file', type=str, default="clip_006.raw", help="The specific frame index to process")
    parser.add_argument('--frame', type=int, default=1000, help="The specific frame index to process")
    args = parser.parse_args()
    
    process_snapshot(args.frame, args.event_file)