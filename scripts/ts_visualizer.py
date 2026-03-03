import argparse
import numpy as np
import cv2
import os
from metavision_core.event_io import EventsIterator

# Configuration
WIDTH, HEIGHT = 1280, 720

# Argument parser
parser = argparse.ArgumentParser(description="Visualize and optionally record events from a .raw file using Time Surface visualization.")
parser.add_argument("input_raw", type=str, help="Path to input .raw file")
parser.add_argument("--delta_t", type=int, default=5000, help="Delta time for EventsIterator (in us, default: 10000)")
parser.add_argument("--output_video", type=str, default=None, help="Optional: Path to save the output video file (e.g., output.avi)")
parser.add_argument("--polarity", type=str, choices=["all", "positive", "negative"], default="all", help="Select polarity to visualize: all, positive (ON), or negative (OFF). Default: all")
parser.add_argument("--color", action="store_true", help="Enable color visualization (Red for ON, Blue for OFF)")
args = parser.parse_args()

# Main variables
raw_path = os.path.expanduser(args.input_raw)
video_path = args.output_video
delta_t = args.delta_t
polarity_mode = args.polarity
use_color = args.color
# Check if file exists
if not os.path.exists(raw_path):
    print(f"Error: Input file not found at {raw_path}")
    exit(1)

# Initialize the Metavision EventsIterator
try:
    mv_iterator = EventsIterator(input_path=raw_path, delta_t=delta_t)
except Exception as e:
    print(f"Error initializing EventsIterator: {e}")
    exit(1)

# Get sensor size from iterator metadata (if available)
try:
    if mv_iterator.get_size() is not None:
        HEIGHT, WIDTH = mv_iterator.get_size()
except:
    # Fallback to default if metadata retrieval fails
    print(f"Warning: Could not retrieve sensor size. Using default: {WIDTH}x{HEIGHT}")

# Visualization and Recording Function
def visualize_time_surface(mv_iterator, width, height, output_video_path, polarity_mode, use_color):
    """
    Iterates through events and visualizes them as a Time Surface.
    Optionally saves the visualization to a video file.
    Filters events based on polarity_mode.
    """
    print(f"Starting visualization for {width}x{height} resolution.")
    print(f"Polarity mode: {polarity_mode}")
    print("Press 'SPACE' to pause/resume. Press 'q' or 'ESC' to quit.")

    video_writer = None
    if output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*'XVID') 
        fps = 30.0 
        
        try:
            video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
            if not video_writer.isOpened():
                raise IOError("Failed to open video writer.")
            print(f"Video will be saved to: {output_video_path} with {fps} FPS.")
        except Exception as e:
            print(f"Error initializing VideoWriter: {e}. Video saving disabled.")
            video_writer = None # Disable saving if initialization fails

    # Persistent buffers
    ts_surface = np.zeros((height, width), dtype=np.uint64)
    # Track the last polarity seen at each pixel for coloring
    pol_surface = np.zeros((height, width), dtype=np.int8) 
    
    FADE_TIME = 10000
    paused = False
    
    for evts in mv_iterator:
        # Filtering logic remains the same
        if polarity_mode == "positive":
            evts = evts[evts['p'] == 1]
        elif polarity_mode == "negative":
            evts = evts[evts['p'] == 0]

        if evts.size > 0:
            x, y, t, p = evts['x'], evts['y'], evts['t'], evts['p']
            
            # Update both timestamp and polarity buffers
            ts_surface[y, x] = t
            pol_surface[y, x] = p

            # Calculate shared intensity map
            current_time = t[-1] 
            time_diff = current_time - ts_surface
            clipped_diff = np.clip(time_diff, 0, FADE_TIME)
            intensity = (1.0 - clipped_diff / FADE_TIME)

            if use_color:
                # Initialize black BGR frame
                display_bgr = np.zeros((height, width, 3), dtype=np.uint8)
                
                # Identify which pixels are ON (1) and OFF (0)
                on_mask = (pol_surface == 1)
                off_mask = (pol_surface == 0)
                
                # Apply intensity to Red channel (index 2) for ON events
                display_bgr[on_mask, 2] = (intensity[on_mask] * 255).astype(np.uint8)
                # Apply intensity to Blue channel (index 0) for OFF events
                display_bgr[off_mask, 0] = (intensity[off_mask] * 255).astype(np.uint8)
            else:
                # Standard Grayscale logic
                display_gray = (intensity * 255).astype(np.uint8)
                display_bgr = cv2.cvtColor(display_gray, cv2.COLOR_GRAY2BGR)
        
        # Display current time and polarity mode in the image
        info_text = f"t: {current_time/1000:.3f} ms | Mode: {polarity_mode}"
        cv2.putText(display_bgr, info_text, (10, height - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        # Write the frame to the video file if writer is active
        if video_writer:
            video_writer.write(display_bgr)
            
        # Display the frame
        cv2.imshow("Event Time Surface Visualization", display_bgr)
        
        # Handle user input
        key = cv2.waitKey(1) 
        
        if key == ord('q') or key == 27: # 'q' or ESC
            break
        elif key == ord(' '): # SPACE
            paused = not paused
            print(f"Playback {'PAUSED' if paused else 'RESUMED'}")
        
        # If paused, wait indefinitely until a resume or quit command is given
        while paused:
            key = cv2.waitKey(100) # Wait longer while paused
            if key == ord(' '):
                paused = not paused
                print("Playback RESUMED")
                break
            elif key == ord('q') or key == 27:
                cv2.destroyAllWindows()
                if video_writer:
                    video_writer.release()
                return

    # Cleanup
    cv2.destroyAllWindows()
    if video_writer:
        video_writer.release()
        print(f"Video saved to {output_video_path}.")
    print("Visualization finished.")


# Execution
if __name__ == "__main__":
    visualize_time_surface(mv_iterator, WIDTH, HEIGHT, video_path, polarity_mode, use_color)