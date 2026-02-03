#!/usr/bin/env python3
"""Generate review images for all GT annotations."""

import json
import cv2
import numpy as np
from pathlib import Path

ANNOTATIONS_DIR = Path(__file__).parent / "annotations"
VIDEO_CACHE = Path(__file__).parent / "video_cache"
OUTPUT_DIR = Path(__file__).parent / "gt_review"
OUTPUT_DIR.mkdir(exist_ok=True)

def draw_template(frame, annotations):
    """Draw template overlay on frame."""
    vis = frame.copy()

    for ann in annotations:
        if not ann.get('isTemplate'):
            continue

        points = ann.get('points', [])
        names = ann.get('templatePoints', [])
        is_gt = ann.get('isGT', False)

        color = (0, 255, 0) if is_gt else (0, 165, 255)  # Green for GT, orange otherwise

        # Draw points
        for i, (pt, name) in enumerate(zip(points, names)):
            x, y = int(pt[0]), int(pt[1])
            cv2.circle(vis, (x, y), 6, color, -1)
            cv2.circle(vis, (x, y), 8, (255, 255, 255), 1)

            # Label
            label = name.replace('corner_', '').replace('_', ' ').upper()[:8]
            cv2.putText(vis, label, (x + 10, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Draw lines connecting corners if we have them
        corner_names = ['corner_tl', 'corner_tr', 'corner_br', 'corner_bl']
        corners = {}
        for pt, name in zip(points, names):
            if name in corner_names:
                corners[name] = (int(pt[0]), int(pt[1]))

        if len(corners) == 4:
            # Draw quadrilateral
            pts = [corners['corner_tl'], corners['corner_tr'],
                   corners['corner_br'], corners['corner_bl']]
            for i in range(4):
                cv2.line(vis, pts[i], pts[(i+1)%4], color, 2)

    return vis

def main():
    print("Generating GT review images...")

    count = 0
    errors = []

    for ann_file in sorted(ANNOTATIONS_DIR.glob('*.json')):
        try:
            with open(ann_file) as f:
                data = json.load(f)

            video_name = data.get('video', ann_file.stem)

            # Find video file
            video_path = VIDEO_CACHE / video_name
            if not video_path.exists():
                video_path = VIDEO_CACHE / (video_name + '.mp4')
            if not video_path.exists():
                # Try without extension
                for ext in ['.mp4', '.avi', '.mov']:
                    test_path = VIDEO_CACHE / (video_name.replace('.mp4', '') + ext)
                    if test_path.exists():
                        video_path = test_path
                        break

            if not video_path.exists():
                continue

            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                continue

            for frame_key, annotations in data.get('frames', {}).items():
                # Check if this frame has GT template
                has_gt = any(a.get('isGT') and a.get('isTemplate') for a in annotations)
                if not has_gt:
                    continue

                frame_num = int(frame_key)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()

                if not ret:
                    errors.append(f"Failed to read {video_name} frame {frame_num}")
                    continue

                # Draw template
                vis = draw_template(frame, annotations)

                # Add info text
                h, w = vis.shape[:2]
                info = f"{video_name[:50]}  Frame: {frame_num}"
                cv2.rectangle(vis, (0, h-30), (w, h), (0, 0, 0), -1)
                cv2.putText(vis, info, (10, h-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # Add index number
                cv2.rectangle(vis, (0, 0), (60, 30), (0, 0, 0), -1)
                cv2.putText(vis, f"#{count+1}", (5, 22),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Save
                short_name = video_name[:40].replace('/', '_').replace(' ', '_')
                out_path = OUTPUT_DIR / f"{count:03d}_{short_name}_f{frame_num}.jpg"
                cv2.imwrite(str(out_path), vis, [cv2.IMWRITE_JPEG_QUALITY, 85])

                count += 1
                print(f"  [{count}] {video_name[:40]}... frame {frame_num}")

            cap.release()

        except Exception as e:
            errors.append(f"Error processing {ann_file.name}: {e}")

    print(f"\n=== Generated {count} GT review images ===")
    print(f"Output: {OUTPUT_DIR}")

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for e in errors[:10]:
            print(f"  - {e}")

    # Also create a contact sheet (grid of all images)
    print("\nCreating contact sheet...")
    create_contact_sheet(OUTPUT_DIR, count)

def create_contact_sheet(output_dir, total):
    """Create a grid of all GT images."""
    images = sorted(output_dir.glob('*.jpg'))[:total]

    if not images:
        return

    # Read first image to get dimensions
    sample = cv2.imread(str(images[0]))
    img_h, img_w = sample.shape[:2]

    # Calculate grid size (aim for roughly square)
    cols = int(np.ceil(np.sqrt(len(images) * img_w / img_h)))
    rows = int(np.ceil(len(images) / cols))

    # Thumbnail size
    thumb_w, thumb_h = 320, 180

    # Create contact sheet
    sheet = np.zeros((rows * thumb_h, cols * thumb_w, 3), dtype=np.uint8)

    for i, img_path in enumerate(images):
        row = i // cols
        col = i % cols

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # Resize to thumbnail
        thumb = cv2.resize(img, (thumb_w, thumb_h))

        y1, y2 = row * thumb_h, (row + 1) * thumb_h
        x1, x2 = col * thumb_w, (col + 1) * thumb_w
        sheet[y1:y2, x1:x2] = thumb

    # Save contact sheet
    sheet_path = output_dir / "_contact_sheet.jpg"
    cv2.imwrite(str(sheet_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"Contact sheet: {sheet_path}")
    print(f"  Grid: {cols}x{rows} = {len(images)} images")

if __name__ == '__main__':
    main()
