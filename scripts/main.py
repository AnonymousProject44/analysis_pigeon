import argparse
import subprocess
import sys
import os

def run_command(command):
    # Execute a shell command and handle potential errors gracefully
    print(f"Executing: {' '.join(command)}")
    result = subprocess.run(command)
    if result.returncode != 0:
        print(f"Error executing command: {' '.join(command)}")
        sys.exit(1)

def main():
    # Setup argument parser for the entire pipeline
    parser = argparse.ArgumentParser(description="Full processing pipeline for Analysis Pigeon")
    parser.add_argument('--left_raw', type=str, required=True, help="Path to the raw event file for the left camera")
    parser.add_argument('--right_raw', type=str, required=True, help="Path to the raw event file for the right camera")
    parser.add_argument('--clip', type=str, required=True, help="Identifier name for the specific clip being processed")
    parser.add_argument('--mode', type=str, default='event_frame', help="Processing mode to use")
    args = parser.parse_args()

    # Define absolute paths for the auxiliary scripts
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tracking_script = os.path.join(script_dir, "bird_tracking.py")
    matching_script = os.path.join(script_dir, "matching_birds.py")
    visualizer_script = os.path.join(script_dir, "stereo_visualizer.py")

    # Execute tracking for the left camera feed
    left_cmd = [sys.executable, tracking_script, args.left_raw, "--camera", "left", "--mode", args.mode]
    run_command(left_cmd)

    # Execute tracking for the right camera feed
    right_cmd = [sys.executable, tracking_script, args.right_raw, "--camera", "right", "--mode", args.mode]
    run_command(right_cmd)

    # Determine the filetype suffix based on the selected mode
    filetype = "ts" if args.mode == 'time_surface' else "evf"

    # Construct the expected paths for the output CSVs
    left_csv = os.path.abspath(os.path.join(script_dir, f"../csv/tracking_{filetype}_{args.clip}_left.csv"))
    right_csv = os.path.abspath(os.path.join(script_dir, f"../csv/tracking_{filetype}_{args.clip}_right.csv"))

    # Execute stereo matching using the newly generated data
    matching_cmd = [
        sys.executable, matching_script,
        left_csv, right_csv,
        "--clip", args.clip,
        "--mode", args.mode
    ]
    run_command(matching_cmd)

    # Execute stereo visualizer to display the matched trajectories
    visualizer_cmd = [
        sys.executable, visualizer_script,
        args.left_raw, args.right_raw,
        "--clip", args.clip,
        "--mode", args.mode
    ]
    run_command(visualizer_cmd)

    print("Pipeline execution completed successfully.")

if __name__ == "__main__":
    main()