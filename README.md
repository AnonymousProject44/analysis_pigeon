# Analysis Pigeon: Advanced Flight Monitoring with Event Cameras

**Analysis Pigeon** is a comprehensive Python-based ecosystem designed for tracking, stereo matching, and biomechanical analysis of pigeons in flight.  

The system leverages **event cameras** to transform raw event streams into accurate three-dimensional trajectory data, velocity measurements, and wingbeat frequency statistics.

---

# Dataset

The raw event recordings are available on OSF: https://osf.io/dur8z/overview?view_only=e77a171b1d3e4d4aaf1ea5ca90618e4c

The scripts expect the dataset laid out as `{date}/{spot}/{Left|Right}/clips/clip_{xxx}.raw`, and flight tags follow the same structure (`{date}_Spot{n}_clip_{xxx}`).

**Extrinsics.** Each recording session has its own stereo extrinsics at `{date}/{spot}/extrinsics.yaml` in the dataset. `matching_birds.py` loads them from the dataset root given by `--birds_dir` (default `~/Events/Birds`), and falls back to `config/calibration/extrinsics.yaml` only if the session file is missing.

**Spot names vs. the paper.** Sequence IDs in the paper number the spots across both days, so the repository's spot names map to them as follows:

| Repository | Paper |
|---|---|
| `20251127_Spot1` | S1 |
| `20260703_Spot1` | S2 |
| `20260703_Spot2` | S3 |
| `20260703_Spot3` | S4 |
| `20260703_Spot4` | S5 |

For example, `20260703_Spot1_clip_004` is flight S2-004 in the paper.

---

# Configuration Files

The system depends on specific configuration files located inside the `config/` directory (extrinsic and intrinsic parameters, and yolo models).

## YOLO Model Weights

- `ts.pt` → Used when running in **Time Surface** mode  
- `evf.pt` → Used when running in **Event Frame** mode  

## Tracking Parameters

- `tracker_params.yaml` → Gate parameters used by `bird_tracking.py`
- `selected_flights.txt` → One flight tag per line (`{date}_Spot{n}_clip_{xxx}`), the `--flights` input of the 3D filtering, plotting, and leadership scripts below

# Scripts Overview and Usage

The pipeline is divided into modular scripts, each responsible for a specific stage. Scripts import shared helpers from `scripts/naming.py` (flight tag parsing) where noted.

---

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
- `--dt` → Delta time in microseconds (default: 5000)
- `--save_csv` → true or false

### Usage
```bash
python scripts/bird_tracking.py path/to/clip.raw --mode time_surface --dt 5000 --save_csv true
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
- `--dt` → Frame delta time in microseconds, must match the tracking (default: 5000)

### Usage
```bash
python scripts/matching_birds.py csv/left.csv csv/right.csv --clip CLIP_ID --mode time_surface
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
python scripts/stereo_visualizer.py path/to/left.raw path/to/right.raw --clip CLIP_ID --mode time_surface
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
python scripts/main.py --left_raw path/to/left.raw --right_raw path/to/right.raw --clip CLIP_ID --mode time_surface
```
---
# Trajectory Curation

The 3D matching CSVs go through an automatic filtering pass. Filtering always backs up the original file to `*.orig.csv` on first run, so it can be safely re-run or reset with `--from_orig`.

## 3D Trajectory Filtering
`scripts/filter_trajectories3d.py`

Drops 3D trajectories that are too short, barely move, or diverge from the local flock direction, and regenerates the corresponding plot after each pass.

### Parameters
- `csvs` → `matching_*.csv` files to filter
- `--flights` → File with one flight tag per line
- `--min_frames` → Minimum trajectory duration in frames (default: 100)
- `--min_disp_frac` → Minimum displacement as a fraction of the clip median (default: 0.2)
- `--max_angle` → Maximum angle in degrees to the local flock direction (default: 60)
- `--no_plot` → Skip regenerating the 3D plot
- `--from_orig` → Reset from the `.orig.csv` backup before filtering

### Usage
```bash
python scripts/filter_trajectories3d.py --flights config/selected_flights.txt
```

## Regenerate 3D Plots
`scripts/replot3d.py`

Rebuilds the 3D trajectory PDF/PNG for one or more flights from their matching CSVs, without re-running filtering (e.g. after changing the plot style).

### Usage
```bash
python scripts/replot3d.py --flights config/selected_flights.txt
```
---
# Leadership Analysis

Implements the leadership analysis of Section V of the paper: time-delayed directional correlation between flight-direction vectors (Nagy et al., 2010), applied to the filtered 3D trajectories.

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