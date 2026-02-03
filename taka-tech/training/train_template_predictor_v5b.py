#!/usr/bin/env python3
"""V5b - 11 key points: 4 corners + 3 center line + 4 penalty box outer corners"""

import argparse
import json
import random
from pathlib import Path

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

IMG_SIZE = 256

POINT_NAMES = [
    'corner_tl', 'corner_tr', 'corner_bl', 'corner_br',
    'center_top', 'center_mid', 'center_bottom',
    'penalty_left_top', 'penalty_left_bottom',
    'penalty_right_top', 'penalty_right_bottom',
]
NUM_POINTS = 11


class KeyPointPredictor(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, NUM_POINTS * 2)
        )
        with torch.no_grad():
            self.head[-1].weight.zero_()
            defaults = [
                0.1, 0.3, 0.9, 0.3, -0.5, 0.9, 1.5, 0.9,
                0.5, 0.3, 0.5, 0.5, 0.5, 0.9,
                0.15, 0.35, 0.15, 0.55,
                0.85, 0.35, 0.85, 0.55
            ]
            self.head[-1].bias.copy_(torch.tensor(defaults))

    def forward(self, x):
        return self.head(self.backbone(x))


class KeyPointDataset(Dataset):
    def __init__(self, ann_dir, video_dir, aug_factor=20):
        self.ann_dir = Path(ann_dir)
        self.video_dir = Path(video_dir)
        self.aug_factor = aug_factor
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((IMG_SIZE, IMG_SIZE))
        ])
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.samples = []
        self._caps = {}
        self._load()
        print(f"Loaded {len(self.samples)} samples")

    def _load(self):
        for f in self.ann_dir.glob('*.json'):
            with open(f) as fp:
                data = json.load(fp)
            vname = data.get('video', f.stem)
            vpath = None
            for ext in ['', '.mp4']:
                p = self.video_dir / (vname + ext)
                if p.exists():
                    vpath = p
                    break
            if not vpath:
                continue

            cap = cv2.VideoCapture(str(vpath))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            for fk, anns in data.get('frames', {}).items():
                if not any(a.get('isGT') and a.get('isTemplate') for a in anns):
                    continue
                pts = {}
                for a in anns:
                    if not a.get('isTemplate'):
                        continue
                    for n, p in zip(a.get('templatePoints', []), a.get('points', [])):
                        if n in POINT_NAMES:
                            pts[n] = p
                if len(pts) < 4:
                    continue

                arr = np.zeros(NUM_POINTS * 2, np.float32)
                mask = np.zeros(NUM_POINTS, np.float32)
                for i, n in enumerate(POINT_NAMES):
                    if n in pts:
                        arr[2*i] = pts[n][0] / w
                        arr[2*i + 1] = pts[n][1] / h
                        mask[i] = 1.0
                self.samples.append({
                    'vpath': str(vpath),
                    'frame': int(fk),
                    'pts': arr,
                    'mask': mask
                })

    def __len__(self):
        return len(self.samples) * self.aug_factor

    def __getitem__(self, idx):
        s = self.samples[idx % len(self.samples)]
        if s['vpath'] not in self._caps:
            self._caps[s['vpath']] = cv2.VideoCapture(s['vpath'])
        cap = self._caps[s['vpath']]
        cap.set(cv2.CAP_PROP_POS_FRAMES, s['frame'])
        ret, frame = cap.read()

        if not ret:
            return {
                'image': torch.zeros(3, IMG_SIZE, IMG_SIZE),
                'pts': torch.zeros(NUM_POINTS * 2),
                'mask': torch.zeros(NUM_POINTS),
                'valid': torch.tensor(0.0)
            }

        img = self.transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        pts = torch.tensor(s['pts'].copy())
        mask = torch.tensor(s['mask'].copy())

        aug = random.randint(0, 3)
        if aug == 1:  # flip
            img = TF.hflip(img)
            pts[0::2] = 1.0 - pts[0::2]
            # Swap left/right
            new_pts = pts.clone()
            new_pts[0:2] = pts[2:4]  # TL <- TR
            new_pts[2:4] = pts[0:2]  # TR <- TL
            new_pts[4:6] = pts[6:8]  # BL <- BR
            new_pts[6:8] = pts[4:6]  # BR <- BL
            new_pts[14:18] = pts[18:22]  # left penalty <- right
            new_pts[18:22] = pts[14:18]  # right penalty <- left
            pts = new_pts
        elif aug == 2:  # color
            img = TF.adjust_brightness(img, random.uniform(0.8, 1.2))
            img = TF.adjust_contrast(img, random.uniform(0.8, 1.2))
        elif aug == 3:  # rotate
            angle = random.uniform(-5, 5)
            img = TF.rotate(img, angle, fill=0)
            rad = np.radians(angle)
            c, s = np.cos(rad), np.sin(rad)
            for i in range(NUM_POINTS):
                x, y = pts[2*i] - 0.5, pts[2*i + 1] - 0.5
                pts[2*i] = x * c - y * s + 0.5
                pts[2*i + 1] = x * s + y * c + 0.5

        return {
            'image': self.normalize(TF.to_tensor(img)),
            'pts': pts,
            'mask': mask,
            'valid': torch.tensor(1.0)
        }


def train(args):
    device = torch.device('mps' if torch.backends.mps.is_available() else
                          'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    ds = KeyPointDataset(args.annotations, args.videos)
    n = len(ds.samples)
    tn = int(0.8 * n)
    idx = list(range(len(ds)))

    tr_loader = DataLoader(
        torch.utils.data.Subset(ds, [i for i in idx if i % n < tn]),
        batch_size=16, shuffle=True, num_workers=0
    )
    va_loader = DataLoader(
        torch.utils.data.Subset(ds, [i for i in idx if i % n >= tn]),
        batch_size=16, num_workers=0
    )
    print(f"Train: {len([i for i in idx if i % n < tn])}, Val: {len([i for i in idx if i % n >= tn])}")

    model = KeyPointPredictor().to(device)
    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    Path(args.output).mkdir(parents=True, exist_ok=True)
    best = float('inf')

    # Weights: bottom corners 2x, center 1.5x, penalty 1x
    wts = torch.tensor([1.0, 1.0, 2.0, 2.0, 1.5, 1.5, 1.5, 1.0, 1.0, 1.0, 1.0], device=device)

    for ep in range(args.epochs):
        model.train()
        tl = 0
        for b in tqdm(tr_loader, desc=f"Epoch {ep+1}/{args.epochs}"):
            im = b['image'].to(device)
            pts = b['pts'].to(device)
            msk = b['mask'].to(device)
            val = b['valid'].to(device)

            opt.zero_grad()
            pred = model(im)
            diff = (pred.view(-1, NUM_POINTS, 2) - pts.view(-1, NUM_POINTS, 2)).abs().sum(2)
            loss = ((diff * wts * msk).sum() / (msk.sum() + 1e-6)) * val.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item()
        tl /= len(tr_loader)
        sched.step()

        model.eval()
        vl = 0
        with torch.no_grad():
            for b in va_loader:
                im = b['image'].to(device)
                pts = b['pts'].to(device)
                msk = b['mask'].to(device)
                pred = model(im)
                diff = (pred.view(-1, NUM_POINTS, 2) - pts.view(-1, NUM_POINTS, 2)).abs().sum(2)
                vl += (diff * wts * msk).sum().item() / (msk.sum().item() + 1e-6)
        vl /= len(va_loader)

        print(f"Epoch {ep+1}: Train={tl:.4f}, Val={vl:.4f}")
        if vl < best:
            best = vl
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_loss': vl,
                'point_names': POINT_NAMES,
                'model_type': 'keypoint_predictor_v5b'
            }, f"{args.output}/template_predictor_v5b_best.pth")
            print(f"  -> Saved (val_loss={vl:.4f})")

    print(f"\nBest val loss: {best:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotations', default='../labeling/annotations')
    parser.add_argument('--videos', default='../labeling/video_cache')
    parser.add_argument('--output', default='models/template_predictor')
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()
    train(args)
