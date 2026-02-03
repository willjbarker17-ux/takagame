#!/bin/bash
# Upload GT annotations and model weights to SoccerNet bucket
# Usage: ./upload_gt_data.sh

set -e

BUCKET="gs://soccernet"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== Uploading GT Data and Model Weights to SoccerNet Bucket ==="
echo "Bucket: $BUCKET"
echo ""

# Find gcloud CLI
GCLOUD=""
if command -v gcloud &> /dev/null; then
    GCLOUD="gcloud"
elif [ -f "$HOME/google-cloud-sdk/bin/gcloud" ]; then
    GCLOUD="$HOME/google-cloud-sdk/bin/gcloud"
elif [ -f "/tmp/google-cloud-sdk/bin/gcloud" ]; then
    GCLOUD="/tmp/google-cloud-sdk/bin/gcloud"
else
    echo "Error: gcloud CLI not found. Install Google Cloud SDK first."
    echo "  https://cloud.google.com/sdk/docs/install"
    exit 1
fi
echo "Using gcloud: $GCLOUD"

# Upload GT annotations (JSON files)
echo "1. Uploading GT annotations..."
GT_DIR="$PROJECT_ROOT/labeling/annotations"
if [ -d "$GT_DIR" ]; then
    # Create a tar.gz of all annotations
    ANNOTATIONS_ARCHIVE="/tmp/gt_annotations.tar.gz"
    tar -czf "$ANNOTATIONS_ARCHIVE" -C "$PROJECT_ROOT/labeling" annotations/

    $GCLOUD storage cp "$ANNOTATIONS_ARCHIVE" "$BUCKET/training_data/gt_annotations.tar.gz"
    rm "$ANNOTATIONS_ARCHIVE"

    echo "   Uploaded $(ls -1 "$GT_DIR"/*.json 2>/dev/null | wc -l | tr -d ' ') annotation files"
else
    echo "   Warning: GT annotations directory not found: $GT_DIR"
fi

# Upload model weights
echo ""
echo "2. Uploading model weights..."
MODEL_DIR="$PROJECT_ROOT/training/models/template_predictor"

if [ -f "$MODEL_DIR/template_predictor_v4_best.pth" ]; then
    echo "   Uploading V4 model (best for corner prediction)..."
    $GCLOUD storage cp "$MODEL_DIR/template_predictor_v4_best.pth" "$BUCKET/models/template_predictor_v4_best.pth"
fi

if [ -f "$MODEL_DIR/template_predictor_v5b_best.pth" ]; then
    echo "   Uploading V5b model (11 key points)..."
    $GCLOUD storage cp "$MODEL_DIR/template_predictor_v5b_best.pth" "$BUCKET/models/template_predictor_v5b_best.pth"
fi

echo ""
echo "=== Upload Complete ==="
echo ""
echo "Files uploaded to:"
echo "  - $BUCKET/training_data/gt_annotations.tar.gz"
echo "  - $BUCKET/models/template_predictor_v4_best.pth"
echo "  - $BUCKET/models/template_predictor_v5b_best.pth"
echo ""
echo "Your friend can download with:"
echo "  ./scripts/download_gt_data.sh"
