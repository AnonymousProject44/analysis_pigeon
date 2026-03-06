# Analysis Pigeon: Advanced Flight Monitoring with Event Cameras

**Analysis Pigeon** is a comprehensive Python-based ecosystem designed for tracking, stereo matching, and biomechanical analysis of pigeons in flight.  

The system leverages **event cameras** to transform raw event streams into accurate three-dimensional trajectory data, velocity measurements, and wingbeat frequency statistics.

---

# Configuration Files

The system depends on specific configuration files located inside the `config/` directory (extrinsic and intrinsic parameters, and yolo models).

## YOLO Model Weights

- `ts.pt` → Used when running in **Time Surface** mode  
- `evf.pt` → Used when running in **Event Frame** mode  

# Scripts Overview and Usage

The pipeline is divided into modular scripts, each responsible for a specific stage.

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
# Tracker Simulation Benchmark

### What it does

- Loads trajectories from `config/trajectories.yaml`
- Animates birds in `pygame`
- Runs multiple trackers in parallel:
  - **MOE (Ours)** – Custom tracker
  - **CSRT**
  - **KCF**
  - **NANO**
- Compares predictions against ground truth
- Exports metrics and plots

### Metrics

- **Precision (px)** → Mean center error  
- **Robustness (%)** → % of frames with error < 50 px  

For fast mode (no rendering):

```python
VISUALIZATION = False
```

### Run

```bash
python benchmark_simulation.py
```

---
# Installation and Requirements

A `requirements.txt` file is provided for dependency installation.

```bash
pip install -r requirements.txt
```

To install the Metavision SDK for Python, please refer to the official [Prophesee Metavision documentation](https://docs.prophesee.ai/stable/installation/index.html) for the exact installation instructions for your operating system.