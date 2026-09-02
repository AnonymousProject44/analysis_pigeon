# Analysis Pigeon: Advanced Flight Monitoring with Event Cameras

**Analysis Pigeon** is a comprehensive Python-based ecosystem designed for tracking, stereo matching, and biomechanical analysis of pigeons in flight.  

The system leverages **event cameras** to transform raw event streams into accurate three-dimensional trajectory data, velocity measurements, and wingbeat frequency statistics.

---

# Configuration Files

The system depends on specific configuration files located inside the `config/` directory (extrinsic and intrinsic parameters, and yolo models).

## YOLO Model Weights

- `ts.pt` → Used when running in **Time Surface** mode  
- `evf.pt` → Used when running in **Event Frame** mode  

## Tracking Parameters

- `tracker_params.yaml` → Gate parameters used by `bird_tracking.py`
- `selected_flights.txt` → One flight tag per line (`{date}_Spot{n}_clip_{xxx}`), the `--flights` input shared by the filtering, editing, and leadership scripts below

# Scripts Overview and Usage

The pipeline is divided into modular scripts, each responsible for a specific stage. Scripts import shared helpers from `scripts/naming.py` (flight tag parsing) where noted.

---

## Detection Visualization  
`scripts/inference_images.py`

A visualization tool to validate YOLO detection performance.  

It:
- Generates **Time Surface** and **Event Frame** representations  
- Overlays detected centroids  

### Usage

```bash
python scripts/inference_images.py path/to/clip.raw --frame TARGET_FRAME
```

## 2D Bird Tracking
`scripts/bird_tracking.py`

Performs two-dimensional object tracking.

Features:
- Detection association
- Velocity filtering
- Trajectory bridging across frames

### Parameters
- `raw_file_path`
- `--mode` → time_surface or event_frame
- `--camera` → cCamera identifier (Left or Right)
- `--dt` → Delta time in microseconds
- `--save_csv` → true or false

### Usage
```bash
python scripts/bird_tracking.py path/to/clip.raw --mode event_frame --dt DELTA_TIME --save_csv true
``` 

## Stereo Matching & 3D Reconstruction
`scripts/matching_birds.py`
Handles stereo reconstruction and 3D triangulation.

It:
- Rectifies 2D coordinates
- Matches trajectories using cost optimization
- Applies vertical constraints
- Correlates velocities
- Computes 3D positions via triangulation

### Parameters
- `left_tracking_csv`
- `right_tracking_csv`
- `--clip` → Clip identifier (e.g. 006)
- `--mode` → Processing mode

### Usage
```bash
python scripts/matching_birds.py csv/left.csv csv/right.csv --clip CLIP_ID --mode event_frame
```

## Stereo Diagnostic Viewer
`scripts/stereo_visualizer.py`

Plays the left/right raw clips side by side and overlays the reconstructed 3D trajectories in an interactive plot, for checking a stereo match by eye.

### Parameters
- `raw_l`, `raw_r` → Left and right raw files
- `--clip` → Clip identifier (e.g. 006)
- `--mode` → time_surface or event_frame
- `--save_video` → Save the annotated playback to `videos/`

### Usage
```bash
python scripts/stereo_visualizer.py path/to/left.raw path/to/right.raw --clip CLIP_ID --mode event_frame
```
---
# Full Pipeline & Biomechanical Analysis
`scripts/main.py`

Integrates the full workflow from tracking files to final statistics.

### Parameters
- `--left_raw` → Path to left raw file
- `--right_raw` → Path to right raw file
- `--clip` → Clip identifier (e.g. 006)
- `--mode` → Processing mode


### Usage
```bash
python scripts/main.py --left_raw path/to/left.raw --right_raw path/to/right.raw --clip CLIP_ID --mode event_frame
```
---
# Trajectory Curation

Both the 2D tracking CSVs and the 3D matching CSVs go through an automatic filtering pass, with an interactive editor available for manual correction. Filtering always backs up the original file to `*.orig.csv` on first run, so it can be safely re-run or reset with `--from_orig`.

## 2D Track Filtering
`scripts/filter_tracks.py`

Drops tracks that are too short, barely move, or diverge from the local flock direction.

### Parameters
- `csvs` → Tracking CSVs to filter
- `--flights` → File with one flight tag per line, both cameras
- `--min_frames` → Minimum track duration in frames (default: 100)
- `--min_disp_frac` → Minimum displacement as a fraction of the clip median (default: 0.2)
- `--max_angle` → Maximum angle in degrees to the local flock direction (default: 60)

### Usage
```bash
python scripts/filter_tracks.py --flights config/selected_flights.txt
```

## 3D Trajectory Filtering
`scripts/filter_trajectories3d.py`

Same filtering logic as above, applied to the stereo-matched 3D trajectories, and regenerates the corresponding plot after each pass.

### Parameters
- `csvs` → `matching_*.csv` files to filter
- `--flights` → File with one flight tag per line
- `--min_frames`, `--min_disp_frac`, `--max_angle` → Same as `filter_tracks.py`
- `--no_plot` → Skip regenerating the 3D plot
- `--from_orig` → Reset from the `.orig.csv` backup before filtering

### Usage
```bash
python scripts/filter_trajectories3d.py --flights config/selected_flights.txt
```

## Interactive 3D Trajectory Editor
`scripts/edit_trajectories3d.py`

Manual select / merge / delete editor for a single flight's 3D trajectories, for correcting what automatic filtering misses.

- `[click]` select trajectory · `[m]` merge mode · `[x]` delete selected · `[u]` undo
- `[s]` save and regenerate plot · `[r]` reset view · `[o]` toggle outlier points · `[q]` quit
- drag to rotate, shift+drag to pan, wheel or +/- to zoom

### Usage
```bash
python scripts/edit_trajectories3d.py 20260703_Spot1_clip_006
```

## Interactive 2D Track Editor
`scripts/track_editor.py`

Two-phase manual editor for a raw clip's 2D tracking CSV: an area-selection sweep to bulk-discard tracks outside the flock, followed by fine-grained point/track merge and delete.

### Parameters
- `raw_file` → Raw clip the CSV was generated from
- `--mode`, `--camera`, `--dt` → Must match the CSV being edited
- `--csv` → Tracking CSV (default: derived from the raw path)
- `--step` → Seconds between selection stops (default: 3.0)
- `--no_select` → Skip the area-selection phase

### Usage
```bash
python scripts/track_editor.py path/to/clip.raw --mode event_frame --dt DELTA_TIME
```

## Regenerate 3D Plots
`scripts/replot3d.py`

Rebuilds the 3D trajectory PDF/PNG for one or more flights from their matching CSVs, without re-running filtering (e.g. after a manual edit).

### Usage
```bash
python scripts/replot3d.py --flights config/selected_flights.txt
```
---
# Leadership Analysis

Implements the leadership-analysis method described in `docs/leadership_method.pdf`: time-delayed directional correlation between flight-direction vectors (Nagy et al., 2010), applied to the filtered 3D trajectories.

## Dynamic Leadership
`scripts/dynamic_leadership.py`

Sliding-window analysis of the flock's leader: whether it stays fixed (static) or changes over time (dynamic), and how close the leader sits to the flock's convex-hull edge.

### Parameters
- `targets` → Flight tags or `matching_*.csv` paths
- `--flights` → File with one flight tag per line
- `--dt` → Frame delta time in microseconds (default: 5000)

### Usage
```bash
python scripts/dynamic_leadership.py --flights config/selected_flights.txt
```

Outputs `leadership_{flight}.pdf` and `leadership_summary.csv` in `leadership/`.

## Leadership Network
`scripts/leadership_network.py`

Builds the full pairwise leader-follower network per flight: hierarchy levels, triangle transitivity, and per-bird lead/follow roles.

### Parameters
- `targets` → Flight tags or `matching_*.csv` paths
- `--flights` → File with one flight tag per line
- `--dt` → Frame delta time in microseconds (default: 5000)
- `--force` → Redo flights that already have a network PDF

### Usage
```bash
python scripts/leadership_network.py --flights config/selected_flights.txt
```

Outputs `network_{flight}.pdf`, `network_summary.csv`, and `network_birds.csv` in `leadership/`.
---
# Installation and Requirements

A `requirements.txt` file is provided for dependency installation.

```bash
pip install -r requirements.txt
```

To install the Metavision SDK for Python, please refer to the official [Prophesee Metavision documentation](https://docs.prophesee.ai/stable/installation/index.html) for the exact installation instructions for your operating system.