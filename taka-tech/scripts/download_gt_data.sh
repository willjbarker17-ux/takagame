#!/bin/bash
# Download GT annotations and model weights from SoccerNet bucket
# Usage: ./download_gt_data.sh [--models-only] [--annotations-only]

set -e

BUCKET="gs://soccernet"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

DOWNLOAD_MODELS=true
DOWNLOAD_ANNOTATIONS=true

# Parse arguments
for arg in "$@"; do
    case $arg in
        --models-only)
            DOWNLOAD_ANNOTATIONS=false
            ;;
        --annotations-only)
            DOWNLOAD_MODELS=false
            ;;
    esac
done

echo "=== Downloading GT Data from SoccerNet Bucket ==="
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

# Download GT annotations
if [ "$DOWNLOAD_ANNOTATIONS" = true ]; then
    echo "1. Downloading GT annotations..."
    ANNOTATIONS_DIR="$PROJECT_ROOT/labeling/annotations"
    mkdir -p "$ANNOTATIONS_DIR"

    ANNOTATIONS_ARCHIVE="/tmp/gt_annotations.tar.gz"
    if $GCLOUD storage cp "$BUCKET/training_data/gt_annotations.tar.gz" "$ANNOTATIONS_ARCHIVE" 2>/dev/null; then
        tar -xzf "$ANNOTATIONS_ARCHIVE" -C "$PROJECT_ROOT/labeling/"
        rm "$ANNOTATIONS_ARCHIVE"
        echo "   Downloaded $(ls -1 "$ANNOTATIONS_DIR"/*.json 2>/dev/null | wc -l | tr -d ' ') annotation files"
    else
        echo "   Warning: GT annotations not found in bucket"
    fi
fi

# Download model weights
if [ "$DOWNLOAD_MODELS" = true ]; then
    echo ""
    echo "2. Downloading model weights..."
    MODEL_DIR="$PROJECT_ROOT/training/models/template_predictor"
    mkdir -p "$MODEL_DIR"

    echo "   Downloading V4 model..."
    if $GCLOUD storage cp "$BUCKET/models/template_predictor_v4_best.pth" "$MODEL_DIR/" 2>/dev/null; then
        echo "   V4 model downloaded ($(du -h "$MODEL_DIR/template_predictor_v4_best.pth" | cut -f1))"
    else
        echo "   Warning: V4 model not found"
    fi

    echo "   Downloading V5b model..."
    if $GCLOUD storage cp "$BUCKET/models/template_predictor_v5b_best.pth" "$MODEL_DIR/" 2>/dev/null; then
        echo "   V5b model downloaded ($(du -h "$MODEL_DIR/template_predictor_v5b_best.pth" | cut -f1))"
    else
        echo "   Warning: V5b model not found"
    fi
fi

echo ""
echo "=== Download Complete ==="
echo ""
echo "To train a new model on this data:"
echo "  cd training"
echo "  python train_template_predictor_v4.py --epochs 100"
