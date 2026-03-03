import numpy as np
import cv2
import os
import argparse
from metavision_core.event_io import EventsIterator

def visualize_event_frame_paper_style(file_path, delta_t=5000, save_video=False, output_video=None, polarity_choice="all"):
    try:
        mv_iterator = EventsIterator(input_path=file_path, delta_t=delta_t)
    except Exception as error:
        print(f"Failed to open file: {error}")
        return

    height, width = mv_iterator.get_size()
    
    # This ensures the window is resizable and large
    window_name = "Event Frame Representation"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)

    video_writer = None
    if save_video and output_video:
        # Create directory if it doesn't exist
        output_folder = os.path.dirname(output_video)
        if output_folder and not os.path.exists(output_folder):
            os.makedirs(output_folder)
            
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 30.0 # Standard FPS for smooth playback
        video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
        
        if not video_writer.isOpened():
            print("Error: Could not open VideoWriter.")

    print(f"Visualizing: {width}x{height} | Polarity: {polarity_choice}")
    print("Press 'Q' to exit, 'SPACE' to pause.")

    for events in mv_iterator:
        # Create a white background (Paper Style)
        display = np.full((height, width, 3), 255, dtype=np.uint8)

        if events.size > 0:
            x = events['x']
            y = events['y']
            p = events['p']

            # Apply Polarity Filter
            if polarity_choice in ["all", "positive"]:
                pos_mask = (p == 1)
                # Blue for Positive (ON)
                display[y[pos_mask], x[pos_mask]] = [255, 0, 0]
            
            if polarity_choice in ["all", "negative"]:
                neg_mask = (p == 0)
                # Red for Negative (OFF)
                display[y[neg_mask], x[neg_mask]] = [0, 0, 255]

        if video_writer is not None:
            video_writer.write(display)

        cv2.imshow(window_name, display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            print("Paused. Press SPACE to resume.")
            while True:
                key_p = cv2.waitKey(1) & 0xFF
                if key_p == ord(' ') or key_p == ord('q'):
                    break
            if key_p == ord('q'):
                break

    if video_writer:
        video_writer.release()
        print(f"Video saved to {output_video}")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Large Event Frame Visualization")
    parser.add_argument("input_raw", type=str, help="Path to input .raw file")
    parser.add_argument("--delta_t", type=int, default=5000, help="Delta time in us")
    parser.add_argument("--output_video", type=str, default=None, help="Path to save output video")
    parser.add_argument("--polarity", type=str, choices=["all", "positive", "negative"], default="all", help="Polarity filter")
    
    args = parser.parse_args()

    visualize_event_frame_paper_style(
        file_path=args.input_raw, 
        delta_t=args.delta_t, 
        save_video=args.output_video is not None, 
        output_video=args.output_video,
        polarity_choice=args.polarity
    )