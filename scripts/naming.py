import os

def extract_session_tag(path):
    """Pull date (YYYYMMDD) and spot id out of a path like .../Birds/20260703/Spot1/Left/clips/clip_006.raw"""
    date, spot = None, None
    for part in os.path.abspath(path).split(os.sep):
        if date is None and part.isdigit() and len(part) == 8:
            date = part
        elif spot is None and part.lower().startswith('spot'):
            spot = part
    return "_".join(p for p in (date, spot) if p)

def build_raw_tag(raw_path):
    """Tag used in output filenames: {date}_{spot}_{raw_stem}, falling back to the raw stem alone."""
    stem = os.path.splitext(os.path.basename(raw_path))[0]
    session = extract_session_tag(raw_path)
    return f"{session}_{stem}" if session else stem

def session_from_tag(tag):
    """Split a raw tag like '20260703_Spot1_clip_006' into (date, spot); missing parts are None."""
    date, spot = None, None
    for part in tag.split('_'):
        if date is None and part.isdigit() and len(part) == 8:
            date = part
        elif spot is None and part.lower().startswith('spot'):
            spot = part
    return date, spot

def tag_from_tracking_csv(csv_path):
    """Recover the raw tag from a tracking CSV name like tracking_evf_{tag}_left.csv."""
    stem = os.path.splitext(os.path.basename(csv_path))[0]
    parts = stem.split('_')
    if len(parts) >= 4 and parts[0] == 'tracking' and parts[-1] in ('left', 'right'):
        return '_'.join(parts[2:-1])
    return None
