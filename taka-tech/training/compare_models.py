#!/usr/bin/env python3
"""Compare V4 (4 corners) vs V5 (18 points) vs V5b (11 key points) models."""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

# Import models
sys.path.insert(0, str(Path(__file__).parent))
from train_template_predictor_v4 import CornerPredictor, IMG_SIZE as V4_IMG_SIZE
from train_template_predictor_v5 import MultiPointPredictor, POINT_NAMES, NUM_POINTS, IMG_SIZE as V5_IMG_SIZE
from train_template_predictor_v5b import KeyPointPredictor, NUM_POINTS as V5B_NUM_POINTS, IMG_SIZE as V5B_IMG_SIZE

MODEL_DIR = Path(__file__).parent / "models/template_predictor"
ANNOTATIONS_DIR = Path(__file__).parent.parent / "labeling/annotations"
VIDEO_CACHE = Path(__file__).parent.parent / "labeling/video_cache"


def load_v4_model():
    path = MODEL_DIR / "template_predictor_v4_best.pth"
    if not path.exists():
        return None
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    model = CornerPredictor(backbone='resnet18', pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint.get('val_loss', 'N/A')


def load_v5_model():
    path = MODEL_DIR / "template_predictor_v5_best.pth"
    if not path.exists():
        return None
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    model = MultiPointPredictor(backbone='resnet18', pretrained=False, num_points=NUM_POINTS)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint.get('val_loss', 'N/A')


def load_v5b_model():
    path = MODEL_DIR / "template_predictor_v5b_best.pth"
    if not path.exists():
        return None
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    model = KeyPointPredictor(pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint.get('val_loss', 'N/A')


def get_transform(img_size):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_gt_points(annotations):
    """Extract all GT points from annotations."""
    points = {}
    for ann in annotations:
        if not ann.get('isTemplate'):
            continue
        for name, pt in zip(ann.get('templatePoints', []), ann.get('points', [])):
            points[name] = pt
    return points


def compute_corner_errors(pred_corners, gt_points, orig_w, orig_h):
    """Compute errors for 4 corners."""
    errors = {}
    corner_names = [('corner_tl', 'TL'), ('corner_tr', 'TR'), ('corner_bl', 'BL'), ('corner_br', 'BR')]

    for gt_name, pred_name in corner_names:
        if gt_name not in gt_points:
            continue

        gt = gt_points[gt_name]
        pred = pred_corners[pred_name]

        dx = gt[0] - pred[0]
        dy = gt[1] - pred[1]
        errors[pred_name] = np.sqrt(dx*dx + dy*dy)

    return errors


def predict_v4(model, frame_rgb, orig_w, orig_h):
    """Get V4 predictions (4 corners)."""
    transform = get_transform(V4_IMG_SIZE)
    img_tensor = transform(frame_rgb).unsqueeze(0)

    with torch.no_grad():
        corners_norm = model(img_tensor).numpy()[0]

    return {
        'TL': (corners_norm[0] * orig_w, corners_norm[1] * orig_h),
        'TR': (corners_norm[2] * orig_w, corners_norm[3] * orig_h),
        'BL': (corners_norm[4] * orig_w, corners_norm[5] * orig_h),
        'BR': (corners_norm[6] * orig_w, corners_norm[7] * orig_h),
    }


def predict_v5(model, frame_rgb, orig_w, orig_h):
    """Get V5 predictions (all points, return only corners for comparison)."""
    transform = get_transform(V5_IMG_SIZE)
    img_tensor = transform(frame_rgb).unsqueeze(0)

    with torch.no_grad():
        points_norm = model(img_tensor).numpy()[0]

    # Extract corners (first 4 points = 8 values)
    return {
        'TL': (points_norm[0] * orig_w, points_norm[1] * orig_h),
        'TR': (points_norm[2] * orig_w, points_norm[3] * orig_h),
        'BL': (points_norm[4] * orig_w, points_norm[5] * orig_h),
        'BR': (points_norm[6] * orig_w, points_norm[7] * orig_h),
    }


def predict_v5b(model, frame_rgb, orig_w, orig_h):
    """Get V5b predictions (11 key points, return only corners for comparison)."""
    transform = get_transform(V5B_IMG_SIZE)
    img_tensor = transform(frame_rgb).unsqueeze(0)

    with torch.no_grad():
        points_norm = model(img_tensor).numpy()[0]

    # Extract corners (first 4 points = 8 values)
    return {
        'TL': (points_norm[0] * orig_w, points_norm[1] * orig_h),
        'TR': (points_norm[2] * orig_w, points_norm[3] * orig_h),
        'BL': (points_norm[4] * orig_w, points_norm[5] * orig_h),
        'BR': (points_norm[6] * orig_w, points_norm[7] * orig_h),
    }


def main():
    print("Loading models...")
    v4_result = load_v4_model()
    v5_result = load_v5_model()
    v5b_result = load_v5b_model()

    v4_model = v5_model = v5b_model = None

    if v4_result:
        v4_model, v4_loss = v4_result
        print(f"V4 loaded (val_loss={v4_loss})")
    else:
        print("V4 model not found")

    if v5_result:
        v5_model, v5_loss = v5_result
        print(f"V5 loaded (val_loss={v5_loss})")
    else:
        print("V5 model not found")

    if v5b_result:
        v5b_model, v5b_loss = v5b_result
        print(f"V5b loaded (val_loss={v5b_loss})")
    else:
        print("V5b model not found")

    if not v4_model and not v5_model and not v5b_model:
        print("No models to compare")
        return

    # Collect results
    v4_errors = []
    v5_errors = []
    v5b_errors = []

    for ann_file in sorted(ANNOTATIONS_DIR.glob('*.json')):
        with open(ann_file) as f:
            data = json.load(f)

        video_name = data.get('video', ann_file.stem)
        frames = data.get('frames', {})

        for frame_key, annotations in frames.items():
            has_gt = any(a.get('isGT', False) and a.get('isTemplate', False) for a in annotations)
            if not has_gt:
                continue

            gt_points = get_gt_points(annotations)
            if not all(f'corner_{c}' in gt_points for c in ['tl', 'tr', 'bl', 'br']):
                continue

            video_path = VIDEO_CACHE / video_name
            if not video_path.exists():
                video_path = VIDEO_CACHE / (video_name + '.mp4')
            if not video_path.exists():
                continue

            cap = cv2.VideoCapture(str(video_path))
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_key))
            ret, frame = cap.read()
            cap.release()

            if not ret:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            orig_h, orig_w = frame.shape[:2]

            if v4_model:
                pred_v4 = predict_v4(v4_model, frame_rgb, orig_w, orig_h)
                errs = compute_corner_errors(pred_v4, gt_points, orig_w, orig_h)
                v4_errors.append(np.mean(list(errs.values())))

            if v5_model:
                pred_v5 = predict_v5(v5_model, frame_rgb, orig_w, orig_h)
                errs = compute_corner_errors(pred_v5, gt_points, orig_w, orig_h)
                v5_errors.append(np.mean(list(errs.values())))

            if v5b_model:
                pred_v5b = predict_v5b(v5b_model, frame_rgb, orig_w, orig_h)
                errs = compute_corner_errors(pred_v5b, gt_points, orig_w, orig_h)
                v5b_errors.append(np.mean(list(errs.values())))

    # Print results
    print(f"\n{'='*60}")
    print("MODEL COMPARISON (Corner Error in pixels)")
    print(f"{'='*60}")
    print(f"{'Model':<25} {'Mean Error':<15} {'Std':<15} {'Samples'}")
    print(f"{'-'*60}")

    results = []
    if v4_errors:
        mean_err = np.mean(v4_errors)
        print(f"{'V4 (4 corners)':<25} {mean_err:>8.0f}px      {np.std(v4_errors):>8.0f}px      {len(v4_errors)}")
        results.append(('V4', mean_err))
    if v5_errors:
        mean_err = np.mean(v5_errors)
        print(f"{'V5 (18 points)':<25} {mean_err:>8.0f}px      {np.std(v5_errors):>8.0f}px      {len(v5_errors)}")
        results.append(('V5', mean_err))
    if v5b_errors:
        mean_err = np.mean(v5b_errors)
        print(f"{'V5b (11 key points)':<25} {mean_err:>8.0f}px      {np.std(v5b_errors):>8.0f}px      {len(v5b_errors)}")
        results.append(('V5b', mean_err))

    if len(results) > 1:
        results.sort(key=lambda x: x[1])
        best_name, best_err = results[0]
        print(f"\n{'='*60}")
        print(f"BEST MODEL: {best_name} ({best_err:.0f}px mean error)")


if __name__ == '__main__':
    main()
