#!/usr/bin/env python3
"""
Template Predictor Training V4 - Direct Corner Prediction with Heavy Augmentation

Key improvements over V3:
1. Direct corner prediction (8 values: 4 corners x 2 coords) instead of homography params
2. Heavy data augmentation (20x more training samples)
3. Weighted loss - bottom corners weighted higher (more variance)
4. Predicts normalized corner positions directly

This should work better with small datasets.
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF

# Image size for training
IMG_SIZE = 256

# Field coordinates (meters)
FIELD_POINTS = {
    'corner_tl': (0, 0),
    'corner_tr': (105, 0),
    'corner_bl': (0, 68),
    'corner_br': (105, 68),
}


class CornerPredictor(nn.Module):
    """Predicts 4 field corners directly in normalized image coordinates."""

    def __init__(self, backbone: str = 'resnet18', pretrained: bool = True):
        super().__init__()

        if backbone == 'resnet18':
            self.backbone = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
            feature_dim = 512
        elif backbone == 'resnet34':
            self.backbone = models.resnet34(weights='IMAGENET1K_V1' if pretrained else None)
            feature_dim = 512
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # Remove final FC layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])

        # Corner prediction head - outputs 8 values (4 corners x 2 coords)
        # Each corner is in normalized coords [-2, 3] to allow corners outside frame
        self.corner_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 8)
        )

        # Initialize to typical soccer broadcast view
        # TL near top-left, TR near top-right, BL/BR outside frame (steep perspective)
        with torch.no_grad():
            self.corner_head[-1].weight.zero_()
            # [TL_x, TL_y, TR_x, TR_y, BL_x, BL_y, BR_x, BR_y]
            # Normalized: 0=left/top edge, 1=right/bottom edge
            self.corner_head[-1].bias.copy_(
                torch.tensor([0.1, 0.3,   # TL - slightly left of frame, 30% down
                              0.9, 0.3,   # TR - slightly right of frame, 30% down
                              -0.5, 0.9,  # BL - outside left, near bottom
                              1.5, 0.9])  # BR - outside right, near bottom
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        corners = self.corner_head(features)
        return corners


def weighted_corner_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Weighted L1 loss - bottom corners weighted 2x since they vary more."""
    # pred/gt shape: [B, 8] = [TL_x, TL_y, TR_x, TR_y, BL_x, BL_y, BR_x, BR_y]
    weights = torch.tensor([1.0, 1.0,  # TL
                           1.0, 1.0,   # TR
                           2.0, 2.0,   # BL (weighted higher)
                           2.0, 2.0],  # BR (weighted higher)
                          device=pred.device)

    loss = torch.abs(pred - gt) * weights
    return loss.mean()


class AugmentedTemplateDataset(Dataset):
    """Dataset with heavy augmentation for small datasets."""

    def __init__(self, annotations_dir: Path, video_cache_dir: Path,
                 image_size: int = IMG_SIZE, augment_factor: int = 20):
        self.annotations_dir = Path(annotations_dir)
        self.video_cache_dir = Path(video_cache_dir)
        self.image_size = image_size
        self.augment_factor = augment_factor

        self.base_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
        ])

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        self.samples = self._load_samples()
        self._video_caps = {}
        print(f"Loaded {len(self.samples)} base samples")
        print(f"With {augment_factor}x augmentation = {len(self.samples) * augment_factor} effective samples")

    def _load_samples(self) -> List[Dict]:
        samples = []

        for ann_file in self.annotations_dir.glob('*.json'):
            with open(ann_file) as f:
                data = json.load(f)

            video_name = data.get('video', ann_file.stem)
            video_path = self._find_video(video_name)

            if not video_path:
                continue

            # Get video dimensions
            cap = cv2.VideoCapture(str(video_path))
            orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            frames = data.get('frames', {})

            for frame_key, annotations in frames.items():
                has_gt = any(a.get('isGT', False) and a.get('isTemplate', False) for a in annotations)

                if not has_gt:
                    continue

                corners = self._extract_corners(annotations)
                if corners is None:
                    continue

                # Normalize corners to [0,1] (can be outside this range)
                corners_norm = corners.copy()
                corners_norm[0::2] /= orig_w  # x coords
                corners_norm[1::2] /= orig_h  # y coords

                samples.append({
                    'video_path': str(video_path),
                    'frame_num': int(frame_key),
                    'corners': corners_norm,  # [TL_x, TL_y, TR_x, TR_y, BL_x, BL_y, BR_x, BR_y]
                    'orig_size': (orig_w, orig_h)
                })

        return samples

    def _find_video(self, video_name: str) -> Optional[Path]:
        for ext in ['', '.mp4']:
            video_path = self.video_cache_dir / (video_name + ext)
            if video_path.exists():
                return video_path
        return None

    def _extract_corners(self, annotations: List[Dict]) -> Optional[np.ndarray]:
        """Extract corner positions from annotations."""
        corner_names = ['corner_tl', 'corner_tr', 'corner_bl', 'corner_br']
        corners = {}

        for ann in annotations:
            if not ann.get('isTemplate', False):
                continue

            template_pts = ann.get('templatePoints', [])
            pixel_pts = ann.get('points', [])

            for name, pt in zip(template_pts, pixel_pts):
                if name in corner_names:
                    corners[name] = pt

        if len(corners) != 4:
            return None

        # Return as flat array: [TL_x, TL_y, TR_x, TR_y, BL_x, BL_y, BR_x, BR_y]
        return np.array([
            corners['corner_tl'][0], corners['corner_tl'][1],
            corners['corner_tr'][0], corners['corner_tr'][1],
            corners['corner_bl'][0], corners['corner_bl'][1],
            corners['corner_br'][0], corners['corner_br'][1],
        ], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.samples) * self.augment_factor

    def _augment(self, image, corners):
        """Apply augmentation to image and adjust corners accordingly."""
        aug_idx = random.randint(0, 4)

        if aug_idx == 0:
            # No augmentation
            pass
        elif aug_idx == 1:
            # Horizontal flip
            image = TF.hflip(image)
            # Flip x coords: new_x = 1 - old_x
            corners = corners.clone()
            corners[0::2] = 1.0 - corners[0::2]
            # Swap left/right corners
            corners = torch.tensor([
                corners[2], corners[3],  # TR -> TL
                corners[0], corners[1],  # TL -> TR
                corners[6], corners[7],  # BR -> BL
                corners[4], corners[5],  # BL -> BR
            ])
        elif aug_idx == 2:
            # Color jitter
            image = TF.adjust_brightness(image, random.uniform(0.8, 1.2))
            image = TF.adjust_contrast(image, random.uniform(0.8, 1.2))
            image = TF.adjust_saturation(image, random.uniform(0.8, 1.2))
        elif aug_idx == 3:
            # Slight rotation (-5 to 5 degrees)
            angle = random.uniform(-5, 5)
            image = TF.rotate(image, angle, fill=0)
            # Approximate corner adjustment (for small angles, this is close enough)
            rad = np.radians(angle)
            cos_a, sin_a = np.cos(rad), np.sin(rad)
            cx, cy = 0.5, 0.5  # Center of normalized coords
            corners_adj = corners.clone()
            for i in range(4):
                x, y = corners[2*i] - cx, corners[2*i+1] - cy
                corners_adj[2*i] = x * cos_a - y * sin_a + cx
                corners_adj[2*i+1] = x * sin_a + y * cos_a + cy
            corners = corners_adj
        elif aug_idx == 4:
            # Random crop (90-100% of image, then resize back)
            scale = random.uniform(0.9, 1.0)
            new_size = int(self.image_size * scale)

            # Random crop position
            max_offset = self.image_size - new_size
            offset_x = random.randint(0, max(0, max_offset))
            offset_y = random.randint(0, max(0, max_offset))

            image = TF.crop(image, offset_y, offset_x, new_size, new_size)
            image = TF.resize(image, (self.image_size, self.image_size))

            # Adjust corners for crop
            corners = corners.clone()
            # Convert from full image to cropped region
            corners[0::2] = (corners[0::2] * self.image_size - offset_x) / new_size
            corners[1::2] = (corners[1::2] * self.image_size - offset_y) / new_size

        return image, corners

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        base_idx = idx % len(self.samples)
        sample = self.samples[base_idx]

        # Load frame
        if sample['video_path'] not in self._video_caps:
            self._video_caps[sample['video_path']] = cv2.VideoCapture(sample['video_path'])

        cap = self._video_caps[sample['video_path']]
        cap.set(cv2.CAP_PROP_POS_FRAMES, sample['frame_num'])
        ret, frame = cap.read()

        if not ret:
            return {
                'image': torch.zeros(3, self.image_size, self.image_size),
                'corners': torch.zeros(8),
                'valid': torch.tensor(0.0)
            }

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = self.base_transform(frame_rgb)
        corners = torch.tensor(sample['corners'], dtype=torch.float32)

        # Apply augmentation
        image, corners = self._augment(image, corners)

        # Convert to tensor and normalize
        image = TF.to_tensor(image)
        image = self.normalize(image)

        return {
            'image': image,
            'corners': corners,
            'valid': torch.tensor(1.0)
        }


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Dataset with heavy augmentation
    dataset = AugmentedTemplateDataset(
        Path(args.annotations),
        Path(args.videos),
        image_size=IMG_SIZE,
        augment_factor=20  # 20x augmentation
    )

    if len(dataset.samples) < 5:
        print(f"Error: Only {len(dataset.samples)} base samples. Need more GT annotations.")
        return

    # Split train/val (on base samples, before augmentation)
    n_base = len(dataset.samples)
    indices = list(range(len(dataset)))

    # Group by base sample
    train_base = int(0.8 * n_base)
    train_indices = [i for i in indices if (i % n_base) < train_base]
    val_indices = [i for i in indices if (i % n_base) >= train_base]

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Model
    model = CornerPredictor(backbone='resnet18', pretrained=True).to(device)

    # Optimizer with higher learning rate (more data from augmentation)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            images = batch['image'].to(device)
            corners_gt = batch['corners'].to(device)
            valid = batch['valid'].to(device)

            optimizer.zero_grad()
            corners_pred = model(images)

            loss = weighted_corner_loss(corners_pred, corners_gt)
            loss = loss * valid.mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        scheduler.step()

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                corners_gt = batch['corners'].to(device)

                corners_pred = model(images)
                loss = weighted_corner_loss(corners_pred, corners_gt)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}: Train={train_loss:.4f}, Val={val_loss:.4f}")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'img_size': IMG_SIZE,
                'model_type': 'corner_predictor_v4'
            }, output_dir / 'template_predictor_v4_best.pth')
            print(f"  -> Saved best model (val_loss={val_loss:.4f})")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotations', default='../labeling/annotations')
    parser.add_argument('--videos', default='../labeling/video_cache')
    parser.add_argument('--output', default='models/template_predictor')
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()

    train(args)
