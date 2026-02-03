#!/usr/bin/env python3
"""
Football Pitch Labeling Tool

A web-based interface for annotating pitch lines on soccer videos.
Uses homography-constrained optical flow with drift correction for camera motion compensation.
"""

import os
import json
import subprocess
import threading
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Tuple, Dict
import base64

import cv2
import numpy as np
from flask import Flask, render_template, jsonify, request

# ============================================================
# GT Annotation Cache for instant template lookup
# ============================================================
# GT Cache: stores LATEST GT points per video (new points override old ones)
# Structure: video_name -> {'frame': ref_frame, 'points': {point_name: [x,y]}}
GT_CACHE: Dict[str, dict] = {}
GT_CACHE_LOCK = threading.Lock()

# ============================================================
# Cloud Sync for GT Annotations (auto-upload to bucket)
# ============================================================
BUCKET_NAME = "gs://soccernet"
SYNC_ENABLED = True
_sync_pending = False
_sync_lock = threading.Lock()

def find_gcloud():
    """Find gcloud CLI path."""
    import shutil
    # Check PATH first
    gcloud = shutil.which('gcloud')
    if gcloud:
        return gcloud
    # Check common locations
    for path in [
        os.path.expanduser('~/google-cloud-sdk/bin/gcloud'),
        '/tmp/google-cloud-sdk/bin/gcloud',
        '/usr/local/bin/gcloud',
    ]:
        if os.path.exists(path):
            return path
    return None

def sync_gt_to_bucket():
    """Sync GT annotations to cloud bucket (background task)."""
    global _sync_pending

    gcloud = find_gcloud()
    if not gcloud:
        print("[sync] gcloud CLI not found, skipping sync")
        return False

    try:
        # Create archive of annotations
        import tarfile
        import tempfile

        with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tmp:
            tmp_path = tmp.name

        with tarfile.open(tmp_path, 'w:gz') as tar:
            tar.add(str(ANNOTATIONS_DIR), arcname='annotations')

        # Upload to bucket
        result = subprocess.run(
            [gcloud, 'storage', 'cp', tmp_path, f'{BUCKET_NAME}/training_data/gt_annotations.tar.gz'],
            capture_output=True,
            text=True,
            timeout=60
        )

        # Clean up
        os.unlink(tmp_path)

        if result.returncode == 0:
            print(f"[sync] GT annotations synced to bucket")
            return True
        else:
            print(f"[sync] Upload failed: {result.stderr}")
            return False

    except Exception as e:
        print(f"[sync] Error: {e}")
        return False

def schedule_sync():
    """Schedule a background sync (debounced - waits 5s for more changes)."""
    global _sync_pending

    if not SYNC_ENABLED:
        return

    with _sync_lock:
        if _sync_pending:
            return  # Already scheduled
        _sync_pending = True

    def delayed_sync():
        global _sync_pending
        time.sleep(5)  # Wait for more potential changes
        with _sync_lock:
            _sync_pending = False
        sync_gt_to_bucket()

    thread = threading.Thread(target=delayed_sync, daemon=True)
    thread.start()

def load_gt_cache():
    """Load GT annotations into cache - only keeps the LATEST labeled points per video.

    When multiple frames have GT labels for the same point, only the most recent
    (highest frame number) is kept. New labels override old ones.
    """
    global GT_CACHE
    annotations_dir = Path(__file__).parent / "annotations"

    cache = {}
    for ann_file in annotations_dir.glob('*.json'):
        try:
            with open(ann_file) as f:
                data = json.load(f)

            video_name = data.get('video', ann_file.stem)

            # Find the LATEST GT frame (highest frame number with GT template)
            latest_frame = -1
            latest_points = {}

            for frame_key, annotations in data.get('frames', {}).items():
                frame_num = int(frame_key)

                # Collect points from ALL GT template annotations in this frame
                # (template is stored as multiple line annotations)
                for ann in annotations:
                    if ann.get('isGT') and ann.get('isTemplate'):
                        # This frame has GT - extract all points
                        for name, pt in zip(ann.get('templatePoints', []), ann.get('points', [])):
                            # Only use this point if it's from a newer frame
                            # or if we haven't seen this point yet
                            if name not in latest_points or frame_num > latest_points[name]['frame']:
                                latest_points[name] = {'pos': pt, 'frame': frame_num}
                                if frame_num > latest_frame:
                                    latest_frame = frame_num

            if latest_points:
                # Convert to simple format: {point_name: [x, y]}
                points = {name: data['pos'] for name, data in latest_points.items()}
                corners = {k: v for k, v in points.items()
                          if k in ['corner_tl', 'corner_tr', 'corner_bl', 'corner_br']}

                if len(corners) == 4:
                    cache[video_name] = {
                        'frame': latest_frame,
                        'corners': corners,
                        'all_points': points
                    }

        except Exception as e:
            print(f"Error loading GT cache from {ann_file}: {e}")

    with GT_CACHE_LOCK:
        GT_CACHE = cache

    print(f"✓ GT Cache loaded: {len(cache)} videos (latest points only, new overrides old)")
    return cache

def get_gt_corners(video_name: str, frame_num: int) -> Optional[dict]:
    """Get GT corners for a video.

    Returns the LATEST labeled corners regardless of frame number.
    New labels always override old ones - no interpolation between old/new.
    """
    with GT_CACHE_LOCK:
        if video_name not in GT_CACHE:
            return None

        video_data = GT_CACHE[video_name]
        return video_data.get('corners')

def _update_gt_cache(video_name: str, frame_num: int, annotations: list):
    """Update GT cache when annotations are saved.

    New points OVERRIDE all previous points for that point type.
    The latest save is always the source of truth.
    """
    global GT_CACHE

    # Find GT template annotation
    gt_ann = None
    for ann in annotations:
        if ann.get('isGT') and ann.get('isTemplate'):
            gt_ann = ann
            break

    if not gt_ann:
        return

    # Extract all points from the new GT annotation
    new_points = {}
    for name, pt in zip(gt_ann.get('templatePoints', []), gt_ann.get('points', [])):
        new_points[name] = pt

    corners = {k: v for k, v in new_points.items()
              if k in ['corner_tl', 'corner_tr', 'corner_bl', 'corner_br']}

    if len(corners) != 4:
        return

    # OVERRIDE: Replace all previous data with new points
    with GT_CACHE_LOCK:
        GT_CACHE[video_name] = {
            'frame': frame_num,
            'corners': corners,
            'all_points': new_points
        }
        print(f"✓ GT Cache OVERRIDE: {video_name} frame {frame_num} (new points replace all old)")

    # Auto-sync GT annotations to cloud bucket
    schedule_sync()


def start_background_training():
    """Start V4 training in background if new GT data exists."""
    training_dir = Path(__file__).parent.parent / 'training'
    model_path = training_dir / 'models' / 'template_predictor' / 'template_predictor_v4_best.pth'
    annotations_dir = Path(__file__).parent / 'annotations'

    # Check if model exists and if annotations are newer
    if model_path.exists():
        model_time = model_path.stat().st_mtime

        # Check if any annotation file is newer than model
        newer_exists = False
        for ann_file in annotations_dir.glob('*.json'):
            if ann_file.stat().st_mtime > model_time:
                newer_exists = True
                break

        if not newer_exists:
            print("✓ V4 model is up-to-date, skipping training")
            return

    print("\n" + "="*60)
    print("STARTING BACKGROUND V4 TRAINING (new GT data detected)")
    print("="*60)

    def run_training():
        try:
            script_path = training_dir / 'train_template_predictor_v4.py'
            if not script_path.exists():
                print("Training script not found")
                return

            process = subprocess.Popen(
                ['python3', str(script_path), '--epochs', '50'],
                cwd=str(training_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            for line in process.stdout:
                line = line.strip()
                if line:
                    # Only print epoch summaries and important info
                    if 'Epoch' in line and 'Train=' in line:
                        print(f"[V4 Training] {line}")
                    elif 'Saved' in line or 'Best' in line or 'Device' in line:
                        print(f"[V4 Training] {line}")

            process.wait()
            if process.returncode == 0:
                print("[V4 Training] ✓ Complete! Model updated.")
                # Reload GT cache after training
                load_gt_cache()
            else:
                print(f"[V4 Training] Failed with code {process.returncode}")
        except Exception as e:
            print(f"[V4 Training] Error: {e}")

    thread = threading.Thread(target=run_training, daemon=True)
    thread.start()

# Try to import improved tracking (falls back to basic if not available)
try:
    from improved_tracking import (
        track_points_optical_flow as improved_track_points,
        propagate_annotations_optical_flow as improved_propagate,
        propagate_annotations_smart,
        get_tracker,
        reset_tracker,
        IMPROVED_TRACKING_AVAILABLE,
        CAMERA_MOTION_AVAILABLE
    )
    print("Using IMPROVED homography-constrained tracking with Kalman filtering")
    if CAMERA_MOTION_AVAILABLE:
        print("  + Camera motion tracking for static field (ORB feature matching)")
except ImportError:
    IMPROVED_TRACKING_AVAILABLE = False
    CAMERA_MOTION_AVAILABLE = False
    print("Using basic Lucas-Kanade optical flow (improved tracking not available)")

app = Flask(__name__)

# Configuration
GCS_BUCKET = "gs://soccernet/wyscout_videos/"
GCLOUD_PATH = "/Users/Will/google-cloud-sdk/bin/gcloud"
ANNOTATIONS_DIR = Path(__file__).parent / "annotations"
ANNOTATIONS_DIR.mkdir(exist_ok=True)

# Persistent local cache for downloaded videos
VIDEO_CACHE_DIR = Path(__file__).parent / "video_cache"
VIDEO_CACHE_DIR.mkdir(exist_ok=True)

# Cache for video list (refreshes every 5 minutes)
_video_list_cache = {'videos': [], 'timestamp': 0}
VIDEO_LIST_CACHE_TTL = 300  # 5 minutes

# Cache for loaded videos and frames
video_cache = {}
frame_cache = {}  # (video, frame_num) -> frame
MAX_FRAME_CACHE = 100

# Semantic pitch line classes (SoccerNet format)
PITCH_CLASSES = [
    "Big rect. left bottom",
    "Big rect. left main",
    "Big rect. left top",
    "Big rect. right bottom",
    "Big rect. right main",
    "Big rect. right top",
    "Small rect. left bottom",
    "Small rect. left main",
    "Small rect. left top",
    "Small rect. right bottom",
    "Small rect. right main",
    "Small rect. right top",
    "Circle central",
    "Circle left",
    "Circle right",
    "Goal left crossbar",
    "Goal left post left",
    "Goal left post right",
    "Goal right crossbar",
    "Goal right post left",
    "Goal right post right",
    "Middle line",
    "Side line left",
    "Side line right",
    "Side line top",
    "Side line bottom",
]


def _auto_learn_from_annotations(video_name: str, frame_num: int, annotations: list):
    """Auto-add GT annotations to feature database for instant learning.

    Called automatically when saving annotations. Only adds if:
    - Annotations have isGT=true AND isTemplate=true
    - At least 4 template points exist (enough for homography)
    - Entry doesn't already exist in database
    """
    try:
        # Check if any annotation has GT template points
        has_gt_template = any(
            ann.get('isGT') and ann.get('isTemplate')
            for ann in annotations
        )

        if not has_gt_template:
            return  # Not GT, skip

        from soccernet_features import (
            compute_homography_from_annotations,
            get_feature_extractor,
            get_feature_database
        )

        # Compute homography
        H = compute_homography_from_annotations(annotations)
        if H is None:
            return  # Not enough points

        # Check if already in database
        db = get_feature_database()
        already_exists = any(
            m.get('video_id') == video_name and m.get('frame_num') == frame_num
            for m in db.metadata
        )

        if already_exists:
            return  # Already learned

        # Get video frame
        video_path = VIDEO_CACHE_DIR / video_name
        if not video_path.exists():
            return

        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return

        # Extract features and add to database
        extractor = get_feature_extractor()
        features = extractor.extract(frame)
        db.add(features, H, video_id=video_name, frame_num=frame_num)

        print(f"Auto-learned: {video_name} frame {frame_num} (database now has {db.size()} entries)")

    except ImportError:
        pass  # SoccerNet features not available
    except Exception as e:
        print(f"Auto-learn error: {e}")


def _refresh_feature_database():
    """Refresh feature database from all annotations on startup."""
    try:
        from soccernet_features import build_database_from_annotations
        print("\nRefreshing SoccerNet feature database...")
        build_database_from_annotations(
            str(ANNOTATIONS_DIR),
            str(VIDEO_CACHE_DIR)
        )
    except ImportError:
        print("SoccerNet features not available (install torch/torchvision)")
    except Exception as e:
        print(f"Feature database refresh error: {e}")


def list_gcs_videos(force_refresh=False):
    """List all videos in GCS bucket with caching."""
    import time
    now = time.time()

    # Return cached list if still valid
    if not force_refresh and _video_list_cache['videos'] and (now - _video_list_cache['timestamp']) < VIDEO_LIST_CACHE_TTL:
        return _video_list_cache['videos']

    try:
        result = subprocess.run(
            [GCLOUD_PATH, "storage", "ls", GCS_BUCKET],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode == 0:
            videos = []
            for line in result.stdout.strip().split('\n'):
                if line.endswith('.mp4'):
                    name = line.replace(GCS_BUCKET, '').strip('/')
                    videos.append(name)
            videos = sorted(videos)
            _video_list_cache['videos'] = videos
            _video_list_cache['timestamp'] = now
            return videos
    except Exception as e:
        print(f"Error listing GCS: {e}")

    # Return cached list even if stale, if available
    return _video_list_cache['videos'] or []


# Track downloads in progress
_download_locks = {}
_download_status = {}  # video_name -> {'status': 'downloading'/'done'/'error', 'progress': 0-100}


def get_cached_video_path(video_name: str) -> Optional[str]:
    """Get path to cached video if it exists."""
    # Sanitize filename for local storage
    safe_name = video_name.replace('/', '_')
    cache_path = VIDEO_CACHE_DIR / safe_name
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return str(cache_path)
    return None


def download_video(video_name: str, background: bool = False) -> Optional[str]:
    """Download video from GCS to local cache."""
    # Check if already cached
    cached_path = get_cached_video_path(video_name)
    if cached_path:
        _download_status[video_name] = {'status': 'done', 'progress': 100}
        return cached_path

    # Sanitize filename
    safe_name = video_name.replace('/', '_')
    cache_path = VIDEO_CACHE_DIR / safe_name

    # Use lock to prevent duplicate downloads
    if video_name not in _download_locks:
        _download_locks[video_name] = threading.Lock()

    def do_download():
        try:
            _download_status[video_name] = {'status': 'downloading', 'progress': 50}

            # Download to temp file first, then rename (atomic)
            temp_path = str(cache_path) + '.tmp'
            gcs_path = GCS_BUCKET + video_name

            result = subprocess.run(
                [GCLOUD_PATH, "storage", "cp", gcs_path, temp_path],
                capture_output=True, text=True, timeout=300
            )

            if result.returncode == 0:
                # Rename to final path
                os.rename(temp_path, str(cache_path))
                _download_status[video_name] = {'status': 'done', 'progress': 100}
                return str(cache_path)
            else:
                print(f"Error downloading video {video_name}: {result.stderr}")
                _download_status[video_name] = {'status': 'error', 'progress': 0, 'error': result.stderr}
                return None

        except Exception as e:
            print(f"Error downloading video {video_name}: {e}")
            _download_status[video_name] = {'status': 'error', 'progress': 0, 'error': str(e)}
            # Clean up partial download
            temp_path = str(cache_path) + '.tmp'
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return None

    if background:
        # Download in background thread
        if _download_locks[video_name].locked():
            return None  # Already downloading

        def bg_task():
            with _download_locks[video_name]:
                do_download()

        thread = threading.Thread(target=bg_task)
        thread.daemon = True
        thread.start()
        return None
    else:
        # Blocking download
        with _download_locks[video_name]:
            return do_download()


def get_video_info(video_path: str) -> dict:
    """Get video metadata."""
    cap = cv2.VideoCapture(video_path)
    info = {
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return info


def get_frame(video_path: str, frame_num: int) -> Optional[np.ndarray]:
    """Extract a specific frame from video with caching."""
    cache_key = (video_path, frame_num)

    if cache_key in frame_cache:
        return frame_cache[cache_key]

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()

    if ret:
        # Manage cache size
        if len(frame_cache) >= MAX_FRAME_CACHE:
            # Remove oldest entries
            keys_to_remove = list(frame_cache.keys())[:20]
            for k in keys_to_remove:
                del frame_cache[k]
        frame_cache[cache_key] = frame
        return frame
    return None


def frame_to_base64(frame: np.ndarray) -> str:
    """Convert frame to base64 encoded JPEG."""
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buffer).decode('utf-8')


def track_points_optical_flow(
    frame1: np.ndarray,
    frame2: np.ndarray,
    points: List[Tuple[float, float]]
) -> List[Tuple[float, float]]:
    """
    Track points from frame1 to frame2 using optical flow.

    If improved tracking is available, uses:
    - Homography-constrained flow (points move together geometrically)
    - Kalman filtering for temporal smoothness
    - Learned drift correction (if model trained)

    Otherwise falls back to basic Lucas-Kanade.
    """
    if not points:
        return []

    # Use improved tracking if available
    if IMPROVED_TRACKING_AVAILABLE:
        return improved_track_points(frame1, frame2, points)

    # Fallback to basic Lucas-Kanade
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    p0 = np.array(points, dtype=np.float32).reshape(-1, 1, 2)

    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    )

    p1, status, error = cv2.calcOpticalFlowPyrLK(gray1, gray2, p0, None, **lk_params)

    if p1 is None:
        return points

    tracked = []
    for i, (new_pt, st) in enumerate(zip(p1, status)):
        if st[0] == 1:
            tracked.append((float(new_pt[0][0]), float(new_pt[0][1])))
        else:
            tracked.append(points[i])

    return tracked


def propagate_annotations_optical_flow(
    frame1: np.ndarray,
    frame2: np.ndarray,
    annotations: List[dict]
) -> List[dict]:
    """
    Propagate all annotations from frame1 to frame2.

    Uses smart propagation that chooses the best method:
    - Template/field annotations: Camera motion tracking (ORB feature matching)
      This is better for static field because all points move together
    - Player/ball annotations: Optical flow tracking

    If improved tracking is available, uses homography-constrained tracking
    which maintains geometric consistency between all points.
    """
    if not annotations:
        return []

    # Use smart propagation if camera motion is available
    # This uses camera motion for templates, optical flow for players/ball
    if CAMERA_MOTION_AVAILABLE:
        return propagate_annotations_smart(frame1, frame2, annotations)

    # Fall back to improved optical flow tracking
    if IMPROVED_TRACKING_AVAILABLE:
        return improved_propagate(frame1, frame2, annotations)

    # Fallback to basic point-by-point tracking
    propagated = []

    for ann in annotations:
        new_ann = ann.copy()
        new_ann['isGT'] = False

        if ann['type'] == 'line':
            points = ann['points']
            tracked_points = track_points_optical_flow(frame1, frame2,
                [(p[0], p[1]) for p in points])
            new_ann['points'] = [[p[0], p[1]] for p in tracked_points]

        elif ann['type'] == 'ellipse':
            cx, cy = ann['center']
            rx, ry = ann['axes']
            angle = ann.get('angle', 0)

            edge_points = []
            for t in [0, 90, 180, 270]:
                rad = np.radians(t + angle)
                x = cx + rx * np.cos(rad)
                y = cy + ry * np.sin(rad)
                edge_points.append((x, y))

            all_points = [(cx, cy)] + edge_points
            tracked = track_points_optical_flow(frame1, frame2, all_points)

            new_center = tracked[0]
            new_ann['center'] = [new_center[0], new_center[1]]

            tracked_edges = tracked[1:]
            if len(tracked_edges) >= 4:
                new_rx = (abs(tracked_edges[0][0] - new_center[0]) +
                         abs(tracked_edges[2][0] - new_center[0])) / 2
                new_ry = (abs(tracked_edges[1][1] - new_center[1]) +
                         abs(tracked_edges[3][1] - new_center[1])) / 2
                new_ann['axes'] = [max(new_rx, 5), max(new_ry, 5)]

        elif ann['type'] == 'point':
            point = ann['point']
            tracked = track_points_optical_flow(frame1, frame2, [(point[0], point[1])])
            new_ann['point'] = [tracked[0][0], tracked[0][1]]

        propagated.append(new_ann)

    return propagated


def load_annotations(video_name: str) -> dict:
    """Load annotations for a video with error handling."""
    ann_file = ANNOTATIONS_DIR / f"{video_name}.json"
    if ann_file.exists():
        try:
            with open(ann_file, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            print(f"[load_annotations] JSON error in {video_name}: {e}")
            # Try to recover by returning empty annotations
            # The corrupted file will be overwritten on next save
            return {'video': video_name, 'frames': {}, 'keyframes': [], '_corrupted': True}
        except Exception as e:
            print(f"[load_annotations] Error loading {video_name}: {e}")
            return {'video': video_name, 'frames': {}, 'keyframes': []}
    return {'video': video_name, 'frames': {}, 'keyframes': []}


# File lock for safe concurrent writes
import fcntl

def save_annotations(video_name: str, annotations: dict):
    """Save annotations for a video with file locking."""
    ann_file = ANNOTATIONS_DIR / f"{video_name}.json"
    annotations['video'] = video_name
    annotations['last_modified'] = datetime.now().isoformat()

    # Remove any recovery flags
    annotations.pop('_corrupted', None)

    # Use atomic write with file locking
    temp_file = ann_file.with_suffix('.json.tmp')
    try:
        with open(temp_file, 'w') as f:
            # Get exclusive lock
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                json.dump(annotations, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        # Atomic rename
        temp_file.rename(ann_file)
    except Exception as e:
        print(f"[save_annotations] Error saving {video_name}: {e}")
        # Clean up temp file if it exists
        if temp_file.exists():
            temp_file.unlink()
        raise


# Routes
@app.route('/')
def index():
    """Main labeling interface."""
    return render_template('index.html', pitch_classes=PITCH_CLASSES)


@app.route('/review')
def gt_review():
    """GT review interface - view and delete bad GT annotations."""
    return render_template('gt_review.html')


@app.route('/api/videos')
def api_videos():
    """List available videos."""
    videos = list_gcs_videos()

    # Add cache status for each video
    video_info = []
    for v in videos:
        cached = get_cached_video_path(v) is not None
        status = _download_status.get(v, {})
        video_info.append({
            'name': v,
            'cached': cached,
            'download_status': status.get('status'),
            'download_progress': status.get('progress', 0)
        })

    return jsonify({'videos': video_info, 'count': len(videos)})


@app.route('/api/video/<path:video_name>/download_status')
def api_download_status(video_name):
    """Check download status for a video."""
    cached = get_cached_video_path(video_name) is not None
    status = _download_status.get(video_name, {})
    return jsonify({
        'video': video_name,
        'cached': cached,
        'status': status.get('status', 'not_started'),
        'progress': status.get('progress', 0),
        'error': status.get('error')
    })


@app.route('/api/video/<path:video_name>/prefetch', methods=['POST'])
def api_prefetch(video_name):
    """Start downloading a video in the background."""
    cached = get_cached_video_path(video_name)
    if cached:
        return jsonify({'status': 'already_cached', 'path': cached})

    # Start background download
    download_video(video_name, background=True)
    return jsonify({'status': 'started', 'message': f'Downloading {video_name} in background'})


@app.route('/api/cache/status')
def api_cache_status():
    """Get cache status - list all cached videos and disk usage."""
    cached_files = list(VIDEO_CACHE_DIR.glob('*.mp4'))
    total_size = sum(f.stat().st_size for f in cached_files)

    return jsonify({
        'cached_count': len(cached_files),
        'total_size_mb': round(total_size / (1024 * 1024), 2),
        'cache_dir': str(VIDEO_CACHE_DIR),
        'cached_videos': [f.name for f in cached_files]
    })


@app.route('/api/cache/clear', methods=['POST'])
def api_cache_clear():
    """Clear the video cache."""
    data = request.json or {}
    video_name = data.get('video_name')

    if video_name:
        # Clear specific video
        safe_name = video_name.replace('/', '_')
        cache_path = VIDEO_CACHE_DIR / safe_name
        if cache_path.exists():
            cache_path.unlink()
            return jsonify({'status': 'cleared', 'video': video_name})
        return jsonify({'status': 'not_found', 'video': video_name})
    else:
        # Clear all
        count = 0
        for f in VIDEO_CACHE_DIR.glob('*.mp4'):
            f.unlink()
            count += 1
        return jsonify({'status': 'cleared', 'count': count})


@app.route('/api/video/<path:video_name>/info')
def api_video_info(video_name):
    """Get video information."""
    video_path = download_video(video_name)
    if not video_path:
        return jsonify({'error': 'Failed to download video'}), 404

    info = get_video_info(video_path)
    info['name'] = video_name

    # Load existing annotations count
    ann = load_annotations(video_name)
    info['annotated_frames'] = len(ann.get('frames', {}))
    info['keyframes'] = ann.get('keyframes', [])

    return jsonify(info)


@app.route('/api/video/<path:video_name>/frame/<int:frame_num>')
def api_get_frame(video_name, frame_num):
    """Get a specific frame as base64 image."""
    video_path = download_video(video_name)
    if not video_path:
        return jsonify({'error': 'Failed to download video'}), 404

    frame = get_frame(video_path, frame_num)
    if frame is None:
        return jsonify({'error': 'Failed to get frame'}), 404

    # Load annotations for this frame
    ann = load_annotations(video_name)
    frame_annotations = ann.get('frames', {}).get(str(frame_num), [])

    return jsonify({
        'frame': frame_to_base64(frame),
        'frame_num': frame_num,
        'annotations': frame_annotations,
        'is_keyframe': frame_num in ann.get('keyframes', [])
    })


@app.route('/api/video/<path:video_name>/annotations')
def api_get_all_annotations(video_name):
    """Get all annotations for a video."""
    ann = load_annotations(video_name)
    return jsonify(ann.get('frames', {}))


@app.route('/api/video/<path:video_name>/annotations/clear', methods=['POST'])
def api_clear_all_annotations(video_name):
    """Clear all annotations for a video."""
    ann_file = ANNOTATIONS_DIR / f"{video_name}.json"
    if ann_file.exists():
        ann_file.unlink()  # Delete the file
    return jsonify({'status': 'cleared', 'video': video_name})


@app.route('/api/annotations/list')
def api_list_annotations():
    """List all annotation files."""
    files = []
    for f in ANNOTATIONS_DIR.glob('*.json'):
        files.append(f.name)
    return jsonify(sorted(files))


@app.route('/api/gt/list')
def api_list_gt_frames():
    """List all GT frames with their annotations."""
    gt_frames = []

    for ann_file in sorted(ANNOTATIONS_DIR.glob('*.json')):
        try:
            with open(ann_file) as f:
                data = json.load(f)

            video_name = data.get('video', ann_file.stem)

            for frame_key, annotations in data.get('frames', {}).items():
                # Combine ALL GT template annotations for this frame
                # (template is stored as multiple line annotations)
                all_points = {}
                all_point_names = []
                has_gt = False

                for ann in annotations:
                    if ann.get('isGT') and ann.get('isTemplate'):
                        has_gt = True
                        point_names = ann.get('templatePoints', [])
                        point_coords = ann.get('points', [])

                        for name, pt in zip(point_names, point_coords):
                            if name not in all_points:  # Avoid duplicates
                                all_points[name] = pt
                                all_point_names.append(name)

                if has_gt:
                    gt_frames.append({
                        'video': video_name,
                        'frame': int(frame_key),
                        'points': all_points,
                        'point_names': all_point_names,
                        'num_points': len(all_points)
                    })

        except Exception as e:
            print(f"Error reading {ann_file}: {e}")

    return jsonify({
        'gt_frames': gt_frames,
        'total': len(gt_frames)
    })


@app.route('/api/gt/delete', methods=['POST'])
def api_delete_gt_frame():
    """Delete a GT annotation from a specific frame."""
    data = request.get_json()
    video_name = data.get('video')
    frame_num = data.get('frame')

    if not video_name or frame_num is None:
        return jsonify({'status': 'error', 'message': 'video and frame required'}), 400

    ann_file = ANNOTATIONS_DIR / f"{video_name}.json"
    if not ann_file.exists():
        return jsonify({'status': 'error', 'message': 'Annotation file not found'}), 404

    try:
        with open(ann_file) as f:
            ann_data = json.load(f)

        frame_key = str(frame_num)
        if frame_key not in ann_data.get('frames', {}):
            return jsonify({'status': 'error', 'message': 'Frame not found'}), 404

        # Remove GT flag from template annotations in this frame
        annotations = ann_data['frames'][frame_key]
        deleted = False
        for ann in annotations:
            if ann.get('isGT') and ann.get('isTemplate'):
                ann['isGT'] = False
                deleted = True

        if not deleted:
            return jsonify({'status': 'error', 'message': 'No GT annotation found'}), 404

        # Save
        with open(ann_file, 'w') as f:
            json.dump(ann_data, f, indent=2)

        # Update GT cache
        load_gt_cache()

        # Sync to cloud
        schedule_sync()

        return jsonify({
            'status': 'success',
            'message': f'Deleted GT from {video_name} frame {frame_num}'
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/annotations/<path:filename>')
def api_get_annotation_file(filename):
    """Get a specific annotation file."""
    ann_file = ANNOTATIONS_DIR / filename
    if not ann_file.exists():
        return jsonify({'error': 'File not found'}), 404
    try:
        with open(ann_file) as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/video/<path:video_name>/frame/<int:frame_num>/annotations', methods=['POST'])
def api_save_frame_annotations(video_name, frame_num):
    """Save annotations for a specific frame."""
    data = request.json
    ann = load_annotations(video_name)

    if 'frames' not in ann:
        ann['frames'] = {}
    if 'keyframes' not in ann:
        ann['keyframes'] = []

    annotations = data.get('annotations', [])

    # Only save template annotations (the 9 field lines)
    # Filter out any extra/debug annotations
    template_annotations = [a for a in annotations if a.get('isTemplate', False)]

    # Validate: reject only truly broken annotations (NaN, infinity, or absurdly extreme)
    # Note: corners 2-3x outside frame is NORMAL for angled soccer views
    def has_broken_points(anns):
        for a in anns:
            for pt in a.get('points', []):
                if not isinstance(pt, (list, tuple)) or len(pt) < 2:
                    return True
                if not np.isfinite(pt[0]) or not np.isfinite(pt[1]):
                    return True
                # Only reject if >50x frame size (clearly broken optical flow)
                if abs(pt[0]) > 64000 or abs(pt[1]) > 36000:
                    return True
        return False

    if has_broken_points(template_annotations):
        return jsonify({
            'status': 'error',
            'message': 'Annotations have invalid points (NaN/infinity). Not saved.'
        }), 400

    if template_annotations:
        ann['frames'][str(frame_num)] = template_annotations
    else:
        # If no template annotations, keep all (might be intentional non-template data)
        ann['frames'][str(frame_num)] = annotations

    # Mark as keyframe if it has annotations and isn't already
    if annotations and frame_num not in ann['keyframes']:
        ann['keyframes'].append(frame_num)
        ann['keyframes'].sort()

    save_annotations(video_name, ann)

    # Auto-learn: if annotations have GT template points, add to feature database
    _auto_learn_from_annotations(video_name, frame_num, annotations)

    # Update GT cache for instant learning
    _update_gt_cache(video_name, frame_num, annotations)

    return jsonify({'status': 'saved', 'frame': frame_num})


@app.route('/api/video/<path:video_name>/propagate', methods=['POST'])
def api_propagate(video_name):
    """
    Propagate annotations from source frame to target frame using optical flow.
    This tracks each annotation point through camera motion.
    """
    data = request.json
    source_frame = data['source_frame']
    target_frame = data['target_frame']

    video_path = download_video(video_name)
    if not video_path:
        return jsonify({'error': 'Failed to download video'}), 404

    # Get source annotations - prefer from request, fallback to disk
    source_annotations = data.get('annotations')
    if not source_annotations:
        ann = load_annotations(video_name)
        source_annotations = ann.get('frames', {}).get(str(source_frame), [])

    if not source_annotations:
        return jsonify({'annotations': [], 'message': 'No source annotations'})

    # For large frame gaps, propagate through intermediate frames
    step = 1 if target_frame > source_frame else -1
    current_frame = source_frame
    current_annotations = source_annotations

    # Propagate frame by frame for accuracy
    frame_step = min(5, abs(target_frame - source_frame))  # Max 5 frame jumps

    while current_frame != target_frame:
        next_frame = current_frame + step * min(frame_step, abs(target_frame - current_frame))

        frame1 = get_frame(video_path, current_frame)
        frame2 = get_frame(video_path, next_frame)

        if frame1 is None or frame2 is None:
            break

        current_annotations = propagate_annotations_optical_flow(
            frame1, frame2, current_annotations
        )
        current_frame = next_frame

    return jsonify({
        'annotations': current_annotations,
        'source_frame': source_frame,
        'target_frame': target_frame
    })


@app.route('/api/video/<path:video_name>/propagate_range', methods=['POST'])
def api_propagate_range(video_name):
    """
    Propagate annotations across a range of frames.
    Used for batch propagation from keyframes.
    """
    data = request.json
    source_frame = data['source_frame']
    start_frame = data.get('start_frame', source_frame)
    end_frame = data['end_frame']

    video_path = download_video(video_name)
    if not video_path:
        return jsonify({'error': 'Failed to download video'}), 404

    # Get source annotations - prefer from request, fallback to disk
    ann = load_annotations(video_name)
    source_annotations = data.get('annotations')
    if not source_annotations:
        source_annotations = ann.get('frames', {}).get(str(source_frame), [])

    if not source_annotations:
        return jsonify({'error': 'No source annotations'}), 400

    # Propagate through range
    results = {str(source_frame): source_annotations}
    current_annotations = source_annotations

    step = 1 if end_frame > start_frame else -1

    for frame_num in range(start_frame + step, end_frame + step, step):
        prev_frame = get_frame(video_path, frame_num - step)
        curr_frame = get_frame(video_path, frame_num)

        if prev_frame is None or curr_frame is None:
            break

        current_annotations = propagate_annotations_optical_flow(
            prev_frame, curr_frame, current_annotations
        )
        results[str(frame_num)] = current_annotations

        # Save each frame
        ann['frames'][str(frame_num)] = current_annotations

    save_annotations(video_name, ann)

    return jsonify({
        'propagated_frames': len(results),
        'start': start_frame,
        'end': end_frame
    })


@app.route('/api/export/<path:video_name>')
def api_export(video_name):
    """Export annotations in SoccerNet JSON format."""
    ann = load_annotations(video_name)

    # Convert to SoccerNet format
    soccernet_format = {
        'video': video_name,
        'frames': {}
    }

    for frame_num, annotations in ann.get('frames', {}).items():
        frame_data = {'original_lines': {}}
        for a in annotations:
            label = a.get('label', 'Unknown')
            if a['type'] == 'line':
                frame_data['original_lines'][label] = [
                    {'x': p[0], 'y': p[1]} for p in a['points']
                ]
            elif a['type'] == 'ellipse':
                # Convert ellipse to points
                cx, cy = a['center']
                rx, ry = a['axes']
                angle = a.get('angle', 0)
                points = []
                for t in np.linspace(0, 2*np.pi, 12):
                    x = cx + rx * np.cos(t) * np.cos(np.radians(angle)) - ry * np.sin(t) * np.sin(np.radians(angle))
                    y = cy + rx * np.cos(t) * np.sin(np.radians(angle)) + ry * np.sin(t) * np.cos(np.radians(angle))
                    points.append({'x': float(x), 'y': float(y)})
                frame_data['original_lines'][label] = points

        soccernet_format['frames'][frame_num] = frame_data

    return jsonify(soccernet_format)


# =============================================================================
# TRACKING CONTROL ENDPOINTS
# =============================================================================

@app.route('/api/tracking/status')
def api_tracking_status():
    """Get current tracking system status."""
    status = {
        'improved_tracking_available': IMPROVED_TRACKING_AVAILABLE,
        'tracking_method': 'homography_constrained' if IMPROVED_TRACKING_AVAILABLE else 'basic_lucas_kanade',
    }

    if IMPROVED_TRACKING_AVAILABLE:
        try:
            tracker = get_tracker()
            status['tracker_stats'] = tracker.get_tracking_stats()
        except Exception as e:
            status['tracker_error'] = str(e)

    return jsonify(status)


@app.route('/api/tracking/reset', methods=['POST'])
def api_tracking_reset():
    """Reset tracker state (call when switching videos)."""
    if IMPROVED_TRACKING_AVAILABLE:
        try:
            reset_tracker()
            return jsonify({'status': 'reset', 'message': 'Tracker state cleared'})
        except Exception as e:
            return jsonify({'status': 'error', 'error': str(e)}), 500
    else:
        return jsonify({'status': 'no_op', 'message': 'Improved tracking not available'})


@app.route('/api/tracking/train', methods=['POST'])
def api_start_training():
    """Start training drift correction model from annotations."""
    import subprocess
    import threading

    # Get options from request
    data = request.get_json() or {}
    gt_only = data.get('gt_only', True)  # Default to GT-only for pure data
    epochs = data.get('epochs', 50)

    def run_training():
        script_path = Path(__file__).parent.parent / 'training' / 'train_drift_correction.py'
        if script_path.exists():
            cmd = [
                'python3', str(script_path),
                '--annotations', str(ANNOTATIONS_DIR),
                '--videos', str(VIDEO_CACHE_DIR),
                '--epochs', str(epochs)
            ]
            if gt_only:
                cmd.append('--gt-only')
            subprocess.run(cmd)

    if IMPROVED_TRACKING_AVAILABLE:
        thread = threading.Thread(target=run_training)
        thread.daemon = True
        thread.start()
        return jsonify({
            'status': 'started',
            'message': f'Training started (GT-only: {gt_only}, epochs: {epochs}). Check console for progress.'
        })
    else:
        return jsonify({
            'status': 'error',
            'message': 'Improved tracking module not available'
        }), 400


@app.route('/api/tracking/predict_template', methods=['POST'])
def api_predict_template():
    """Predict template position using GT cache (instant), V4 neural network, or fallback.

    Priority: 1) GT cache (instant learning), 2) V4 neural network, 3) SoccerNet similarity
    """
    try:
        # Get frame data from request
        data = request.get_json()

        # Check GT cache first for instant learning
        video_name = data.get('video_name')
        frame_num = data.get('frame_num', 0)

        if video_name:
            gt_corners = get_gt_corners(video_name, frame_num)
            if gt_corners:
                # Return GT corners directly
                corners_list = [
                    [gt_corners['corner_tl'][0], gt_corners['corner_tl'][1]],
                    [gt_corners['corner_tr'][0], gt_corners['corner_tr'][1]],
                    [gt_corners['corner_bl'][0], gt_corners['corner_bl'][1]],
                    [gt_corners['corner_br'][0], gt_corners['corner_br'][1]],
                ]
                return jsonify({
                    'status': 'success',
                    'method': 'gt_cache',
                    'predicted_corners': corners_list,
                    'message': f'Using GT from cache (video: {video_name}, frame: {frame_num})'
                })
        if not data or 'frame_base64' not in data:
            return jsonify({'status': 'error', 'message': 'No frame data provided'}), 400

        # Decode base64 image
        frame_data = base64.b64decode(data['frame_base64'].split(',')[1] if ',' in data['frame_base64'] else data['frame_base64'])
        nparr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({'status': 'error', 'message': 'Failed to decode frame'}), 400

        orig_h, orig_w = frame.shape[:2]

        # Get optional video name to exclude from matching (leave-one-out)
        exclude_video = data.get('video_name')

        # Try V4 neural network model first (direct corner prediction - best accuracy)
        model_path_v4 = Path(__file__).parent.parent / 'training' / 'models' / 'template_predictor' / 'template_predictor_v4_best.pth'

        if model_path_v4.exists():
            try:
                import torch
                import sys
                sys.path.insert(0, str(Path(__file__).parent.parent / 'training'))

                from train_template_predictor_v4 import CornerPredictor, IMG_SIZE
                import torchvision.transforms as transforms

                device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

                checkpoint = torch.load(model_path_v4, map_location=device, weights_only=False)
                model = CornerPredictor(backbone='resnet18', pretrained=False)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.to(device)
                model.eval()

                transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((IMG_SIZE, IMG_SIZE)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_tensor = transform(frame_rgb).unsqueeze(0).to(device)

                with torch.no_grad():
                    corners_norm = model(img_tensor).cpu().numpy()[0]

                # Convert normalized corners to pixel coordinates
                # Order: TL, TR, BL, BR (x, y pairs)
                img_corners = np.array([
                    [corners_norm[0] * orig_w, corners_norm[1] * orig_h],  # TL
                    [corners_norm[2] * orig_w, corners_norm[3] * orig_h],  # TR
                    [corners_norm[4] * orig_w, corners_norm[5] * orig_h],  # BL
                    [corners_norm[6] * orig_w, corners_norm[7] * orig_h],  # BR
                ], dtype=np.float32)

                # Field corners in standard coordinates (105m x 68m field)
                FIELD_WIDTH = 1050  # 105m in decimeters
                FIELD_HEIGHT = 680  # 68m in decimeters
                field_corners = np.array([
                    [0, 0],                      # TL
                    [FIELD_WIDTH, 0],            # TR
                    [0, FIELD_HEIGHT],           # BL
                    [FIELD_WIDTH, FIELD_HEIGHT], # BR
                ], dtype=np.float32)

                # Compute homography from 4 point correspondences
                H, _ = cv2.findHomography(img_corners, field_corners)

                if H is not None:
                    return jsonify({
                        'status': 'success',
                        'method': 'neural_network_v4',
                        'homography': H.tolist(),
                        'predicted_corners': img_corners.tolist()
                    })
            except Exception as e:
                print(f"V4 model prediction error: {e}")
                import traceback
                traceback.print_exc()

        # Fallback to V3 neural network model (homography parameters)
        model_path_v3 = Path(__file__).parent.parent / 'training' / 'models' / 'template_predictor' / 'template_predictor_v3_best.pth'

        if model_path_v3.exists():
            try:
                import torch
                import sys
                sys.path.insert(0, str(Path(__file__).parent.parent / 'training'))

                from train_template_predictor_v3 import TemplatePredictor, FIELD_WIDTH, FIELD_HEIGHT, IMG_SIZE
                import torchvision.transforms as transforms

                device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

                checkpoint = torch.load(model_path_v3, map_location=device, weights_only=False)
                model = TemplatePredictor(backbone='resnet18', pretrained=False)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.to(device)
                model.eval()

                transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((IMG_SIZE, IMG_SIZE)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_tensor = transform(frame_rgb).unsqueeze(0).to(device)

                with torch.no_grad():
                    h_params = model(img_tensor).cpu().numpy()[0]

                # H_norm maps normalized_image [0,1] -> normalized_field [0,1]
                H_norm = np.array([
                    [h_params[0], h_params[1], h_params[2]],
                    [h_params[3], h_params[4], h_params[5]],
                    [h_params[6], h_params[7], 1.0]
                ], dtype=np.float32)

                # Convert to original coordinates:
                # H_orig = S_field @ H_norm @ S_img_inv
                # where S_img normalizes image coords, S_field denormalizes field coords
                S_img_inv = np.array([
                    [1.0/orig_w, 0, 0],
                    [0, 1.0/orig_h, 0],
                    [0, 0, 1]
                ], dtype=np.float32)

                S_field = np.array([
                    [FIELD_WIDTH, 0, 0],
                    [0, FIELD_HEIGHT, 0],
                    [0, 0, 1]
                ], dtype=np.float32)

                H_orig = S_field @ H_norm @ S_img_inv
                H_orig = H_orig / (H_orig[2, 2] + 1e-10)

                return jsonify({
                    'status': 'success',
                    'method': 'neural_network_v3',
                    'homography': H_orig.tolist()
                })
            except Exception as e:
                print(f"V3 model prediction error: {e}")
                import traceback
                traceback.print_exc()

        # Try field line detection (works well for partial pitch views)
        try:
            from field_line_detector import detect_field_from_lines, get_field_line_detector

            detector = get_field_line_detector()
            H, matched_lines, all_lines = detector.detect_and_compute_homography(frame)

            if H is not None and len(matched_lines) >= 3:
                corners = detector.get_field_corners_from_homography(H)
                corners_list = [
                    [corners['corner_tl'][0], corners['corner_tl'][1]],
                    [corners['corner_tr'][0], corners['corner_tr'][1]],
                    [corners['corner_bl'][0], corners['corner_bl'][1]],
                    [corners['corner_br'][0], corners['corner_br'][1]],
                ]
                print(f"✓ Line detection: matched {len(matched_lines)} lines: {list(matched_lines.keys())}")
                return jsonify({
                    'status': 'success',
                    'method': 'line_detection',
                    'predicted_corners': corners_list,
                    'homography': H.tolist(),
                    'matched_lines': list(matched_lines.keys()),
                    'num_lines_detected': len(all_lines)
                })
        except Exception as e:
            print(f"Line detection error: {e}")

        # Fallback to SoccerNet similarity
        try:
            from soccernet_features import predict_template, get_feature_database

            db = get_feature_database()
            if db.size() > 0:
                result = predict_template(frame, exclude_video=exclude_video)

                if result and result['confidence'] > 0.5:
                    return jsonify({
                        'status': 'success',
                        'method': 'soccernet_similarity',
                        'homography': result['homography'],
                        'confidence': result['confidence'],
                        'similar_video': result['similar_video'],
                        'similar_frame': result['similar_frame'],
                        'database_size': db.size()
                    })
        except ImportError:
            pass
        except Exception as e:
            print(f"SoccerNet prediction error: {e}")

        return jsonify({
            'status': 'no_model',
            'message': 'No template prediction available. Annotate more frames to build database.'
        }), 404

    except Exception as e:
        import traceback
        return jsonify({
            'status': 'error',
            'message': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/tracking/detect_lines', methods=['POST'])
def api_detect_lines():
    """Detect field lines in a frame and return visualization.

    Useful for partial pitch views where corners are off-screen.
    Detects white lines on green grass and matches to field template.
    """
    try:
        data = request.get_json()
        if not data or 'frame_base64' not in data:
            return jsonify({'status': 'error', 'message': 'No frame data provided'}), 400

        # Decode frame
        frame_data = base64.b64decode(data['frame_base64'].split(',')[1] if ',' in data['frame_base64'] else data['frame_base64'])
        nparr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({'status': 'error', 'message': 'Failed to decode frame'}), 400

        from field_line_detector import get_field_line_detector

        detector = get_field_line_detector()
        H, matched_lines, all_lines = detector.detect_and_compute_homography(frame)

        result = {
            'status': 'success',
            'num_lines_detected': int(len(all_lines)),
            'matched_lines': list(matched_lines.keys()) if matched_lines else [],
            'lines': [
                {
                    'x1': float(l.x1), 'y1': float(l.y1), 'x2': float(l.x2), 'y2': float(l.y2),
                    'angle': float(l.angle), 'length': float(l.length),
                    'is_horizontal': True if l.is_horizontal else False,
                    'is_vertical': True if l.is_vertical else False
                }
                for l in all_lines
            ]
        }

        if H is not None:
            corners = detector.get_field_corners_from_homography(H)
            result['predicted_corners'] = [
                [float(corners['corner_tl'][0]), float(corners['corner_tl'][1])],
                [float(corners['corner_tr'][0]), float(corners['corner_tr'][1])],
                [float(corners['corner_bl'][0]), float(corners['corner_bl'][1])],
                [float(corners['corner_br'][0]), float(corners['corner_br'][1])],
            ]
            result['homography'] = [[float(x) for x in row] for row in H.tolist()]

        # Create visualization
        if data.get('include_visualization', False):
            vis = detector.visualize(frame, all_lines, matched_lines, H)
            _, buffer = cv2.imencode('.jpg', vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
            result['visualization_base64'] = 'data:image/jpeg;base64,' + base64.b64encode(buffer).decode()

        return jsonify(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/tracking/learn_template', methods=['POST'])
def api_learn_template():
    """Add current frame's GT annotations to the SoccerNet feature database.

    This enables instant learning - each annotated frame improves future predictions.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'status': 'error', 'message': 'No data provided'}), 400

        video_name = data.get('video_name')
        frame_num = data.get('frame_num')

        if not video_name:
            return jsonify({'status': 'error', 'message': 'video_name required'}), 400

        # Load annotations
        ann = load_annotations(video_name)
        frame_key = str(frame_num) if frame_num is not None else '0'

        if frame_key not in ann.get('frames', {}):
            return jsonify({'status': 'error', 'message': f'No annotations for frame {frame_key}'}), 400

        annotations = ann['frames'][frame_key]

        # Compute homography from annotations
        from soccernet_features import (
            compute_homography_from_annotations,
            get_feature_extractor,
            get_feature_database
        )

        H = compute_homography_from_annotations(annotations)
        if H is None:
            return jsonify({
                'status': 'error',
                'message': 'Need at least 4 template points (isTemplate=true) to compute homography'
            }), 400

        # Get video frame
        video_path = VIDEO_CACHE_DIR / video_name
        if not video_path.exists():
            return jsonify({'status': 'error', 'message': f'Video not found: {video_name}'}), 404

        cap = cv2.VideoCapture(str(video_path))
        if frame_num:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_num))
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return jsonify({'status': 'error', 'message': 'Failed to read frame'}), 500

        # Extract features and add to database
        extractor = get_feature_extractor()
        db = get_feature_database()

        features = extractor.extract(frame)

        # Check if already exists
        entry_id = f"{video_name}:{frame_key}"
        already_exists = any(
            m.get('video_id') == video_name and str(m.get('frame_num')) == frame_key
            for m in db.metadata
        )

        if already_exists:
            return jsonify({
                'status': 'already_exists',
                'message': f'Frame {frame_key} of {video_name} already in database',
                'database_size': db.size()
            })

        db.add(features, H, video_id=video_name, frame_num=int(frame_key))

        return jsonify({
            'status': 'success',
            'message': f'Added frame {frame_key} to feature database',
            'database_size': db.size()
        })

    except ImportError:
        return jsonify({
            'status': 'error',
            'message': 'SoccerNet features not available. Run: python soccernet_features.py build'
        }), 500
    except Exception as e:
        import traceback
        return jsonify({
            'status': 'error',
            'message': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/tracking/database_stats')
def api_database_stats():
    """Get statistics about the feature database."""
    try:
        from soccernet_features import get_feature_database
        db = get_feature_database()

        # Count unique videos
        unique_videos = set(m.get('video_id') for m in db.metadata if m.get('video_id'))

        return jsonify({
            'status': 'success',
            'total_entries': db.size(),
            'unique_videos': len(unique_videos),
            'entries': [
                {'video': m.get('video_id'), 'frame': m.get('frame_num')}
                for m in db.metadata
            ]
        })
    except ImportError:
        return jsonify({
            'status': 'error',
            'message': 'SoccerNet features not available'
        }), 500
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


if __name__ == '__main__':
    print("Starting Football Pitch Labeling Tool...")
    print(f"Annotations saved to: {ANNOTATIONS_DIR}")

    # Load GT cache for instant template lookup
    load_gt_cache()

    # Start background V4 training if new GT data exists
    start_background_training()

    # Auto-refresh feature database on startup
    _refresh_feature_database()

    if IMPROVED_TRACKING_AVAILABLE:
        print("\n✓ IMPROVED TRACKING ENABLED:")
        print("  - Homography-constrained optical flow")
        print("  - Kalman filtering for temporal smoothness")
        print("  - Learned drift correction (if model trained)")
    else:
        print("\n⚠ Using basic Lucas-Kanade optical flow")
        print("  Install improved tracking for better results")

    print("\n✓ SOCCERNET TEMPLATE PREDICTION:")
    print("  - Auto-learns when you save GT annotations")
    print("  - Database refreshes on startup")
    print("  - KNN similarity matching for initial field position")

    if SYNC_ENABLED and find_gcloud():
        print("\n✓ CLOUD SYNC ENABLED:")
        print(f"  - Auto-uploads GT to {BUCKET_NAME}")
        print("  - Syncs 5s after each GT save (debounced)")
        print("  - Your friend can download with: ./scripts/download_gt_data.sh")
    else:
        print("\n⚠ Cloud sync disabled (gcloud not found)")

    print("\nAnnotations will 'stick' to field positions across frames\n")
    app.run(debug=True, host='0.0.0.0', port=5050)
