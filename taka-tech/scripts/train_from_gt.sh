#!/bin/bash
# Train template predictor models from GT annotations
# Usage: ./train_from_gt.sh [v4|v5b|all]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TRAINING_DIR="$PROJECT_ROOT/training"
ANNOTATIONS_DIR="$PROJECT_ROOT/labeling/annotations"
VIDEO_CACHE_DIR="$PROJECT_ROOT/labeling/video_cache"

MODEL=${1:-v4}  # Default to v4

echo "=== Template Predictor Training ==="
echo ""

# Check if annotations exist
if [ ! -d "$ANNOTATIONS_DIR" ] || [ -z "$(ls -A "$ANNOTATIONS_DIR"/*.json 2>/dev/null)" ]; then
    echo "Error: No GT annotations found in $ANNOTATIONS_DIR"
    echo "Run ./scripts/download_gt_data.sh first"
    exit 1
fi

# Check if videos exist
if [ ! -d "$VIDEO_CACHE_DIR" ] || [ -z "$(ls -A "$VIDEO_CACHE_DIR"/*.mp4 2>/dev/null)" ]; then
    echo "Warning: No videos found in $VIDEO_CACHE_DIR"
    echo "You need videos to train. Download them first or link your video directory."
    exit 1
fi

# Count GT samples
GT_COUNT=$(grep -l '"isGT": true' "$ANNOTATIONS_DIR"/*.json 2>/dev/null | wc -l | tr -d ' ')
echo "Found $GT_COUNT annotation files with GT data"
echo ""

cd "$TRAINING_DIR"

case $MODEL in
    v4)
        echo "Training V4 model (4 corners - recommended for most use cases)..."
        python3 train_template_predictor_v4.py \
            --annotations "$ANNOTATIONS_DIR" \
            --videos "$VIDEO_CACHE_DIR" \
            --epochs 100 \
            --output models/template_predictor
        ;;
    v5b)
        echo "Training V5b model (11 key points)..."
        python3 train_template_predictor_v5b.py \
            --annotations "$ANNOTATIONS_DIR" \
            --videos "$VIDEO_CACHE_DIR" \
            --epochs 100 \
            --output models/template_predictor
        ;;
    all)
        echo "Training all models..."
        echo ""
        echo "--- V4 (4 corners) ---"
        python3 train_template_predictor_v4.py \
            --annotations "$ANNOTATIONS_DIR" \
            --videos "$VIDEO_CACHE_DIR" \
            --epochs 100 \
            --output models/template_predictor

        echo ""
        echo "--- V5b (11 key points) ---"
        python3 train_template_predictor_v5b.py \
            --annotations "$ANNOTATIONS_DIR" \
            --videos "$VIDEO_CACHE_DIR" \
            --epochs 100 \
            --output models/template_predictor
        ;;
    *)
        echo "Unknown model: $MODEL"
        echo "Usage: $0 [v4|v5b|all]"
        exit 1
        ;;
esac

echo ""
echo "=== Training Complete ==="
echo "Models saved to: $TRAINING_DIR/models/template_predictor/"
