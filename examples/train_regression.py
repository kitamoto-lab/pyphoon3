"""
Intensity regression benchmark for the Digital Typhoon Dataset V3 using pyphoon3.

Reproduces the benchmark protocol of the V3 paper:
  - ResNet-18 / ResNet-50 (ImageNet weights), first conv replaced to accept the input channels
  - MSE loss on the raw label (pressure in hPa or wind in kt)
  - Adam, lr 1e-4, batch size 16, 50 epochs
  - SS-L2 split: the last two seasons (2024, 2025) are the evaluation set, all earlier seasons are training
  - images clipped to [170, 300] and scaled to [0, 1]
  - image types: full (512x512), resized (224x224, bilinear) or cropped (224x224 center crop)
  - the reported metric is the evaluation RMSE at the end of the last epoch

Examples:
  # WP-1, pressure, ResNet-18, full images (Table 7 of the paper)
  python train_regression.py --data-root /path/to/V3-2025 --basin WP --channels 1 --label pressure

  # Multi-channel (intersection of channels 1, 2 and 3), WP
  python train_regression.py --data-root /path/to/V3-2025 --basin WP --channels 1 2 3
"""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.models import resnet18, resnet50
from torchvision.transforms.functional import center_crop

from pyphoon3.DigitalTyphoonDataset import DigitalTyphoonDataset

STANDARDIZE_RANGE = (170.0, 300.0)


def build_dataset(data_root, basin, channels, label):
    """Single channel: one directory. Multi-channel: pyphoon3 keeps only observation times where all channels exist."""
    dirs = [f'{data_root}/{c}/{basin}' for c in channels]
    if len(dirs) == 1:
        return DigitalTyphoonDataset(f'{dirs[0]}/image/', f'{dirs[0]}/metadata/', f'{dirs[0]}/metadata.json', label)
    return DigitalTyphoonDataset('', '', '', label,
                                 image_dirs=[f'{d}/image/' for d in dirs],
                                 metadata_dirs=[f'{d}/metadata/' for d in dirs],
                                 metadata_jsons=[f'{d}/metadata.json' for d in dirs])


def season_split(dataset, eval_seasons):
    """SS-L<n> split: evaluation = images from eval_seasons, training = images from every other season."""
    eval_idx = dataset.images_from_seasons(eval_seasons).indices
    train_seasons = [s for s in dataset.get_seasons() if s not in eval_seasons]
    train_idx = dataset.images_from_seasons(train_seasons).indices
    return Subset(dataset, train_idx), Subset(dataset, eval_idx)


def build_model(name, in_channels):
    model = {'resnet18': resnet18, 'resnet50': resnet50}[name](weights='DEFAULT')
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model


def preprocess(images, image_type):
    """images: (B, H, W) or (B, C, H, W) brightness temperatures -> (B, C, H', W') in [0, 1]."""
    x = images.float()
    if x.dim() == 3:
        x = x.unsqueeze(1)
    lo, hi = STANDARDIZE_RANGE
    x = (x.clamp(lo, hi) - lo) / (hi - lo)
    if image_type == 'resized':
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
    elif image_type == 'cropped':
        x = center_crop(x, [224, 224])
    return x


def evaluate(model, loader, image_type, device):
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for images, labels in loader:
            preds.append(model(preprocess(images.to(device), image_type)).squeeze(1).cpu())
            truths.append(labels.float())
    preds, truths = torch.cat(preds), torch.cat(truths)
    return torch.sqrt(torch.mean((preds - truths) ** 2)).item()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-root', required=True, help='dataset root containing the channel directories 1/ 2/ 3/ 4/')
    p.add_argument('--basin', default='WP', choices=['WP', 'AU'])
    p.add_argument('--channels', type=int, nargs='+', default=[1])
    p.add_argument('--label', default='pressure', choices=['pressure', 'wind'])
    p.add_argument('--model', default='resnet18', choices=['resnet18', 'resnet50'])
    p.add_argument('--image-type', default='full', choices=['full', 'resized', 'cropped'])
    p.add_argument('--eval-seasons', type=int, nargs='+', default=[2024, 2025])
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--num-workers', type=int, default=8)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--out', default='result.json', help='per-epoch log (JSON)')
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset = build_dataset(args.data_root, args.basin, args.channels, args.label)
    train_set, eval_set = season_split(dataset, args.eval_seasons)
    print(f'{args.basin}-{"+".join(map(str, args.channels))}: {len(dataset)} images, '
          f'train {len(train_set)}, eval {len(eval_set)} (seasons {args.eval_seasons})', flush=True)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, persistent_workers=True)
    eval_loader = DataLoader(eval_set, batch_size=64, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True, persistent_workers=True)

    model = build_model(args.model, len(args.channels)).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    log = []
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        model.train()
        sq_err, n = 0.0, 0
        for images, labels in train_loader:
            labels = labels.float().unsqueeze(1).to(args.device)
            outputs = model(preprocess(images.to(args.device), args.image_type))
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sq_err += loss.item() * len(labels)
            n += len(labels)
        eval_rmse = evaluate(model, eval_loader, args.image_type, args.device)
        log.append({'epoch': epoch, 'train_rmse': (sq_err / n) ** 0.5, 'eval_rmse': eval_rmse,
                    'seconds': time.time() - start})
        print(json.dumps(log[-1]), flush=True)
        with open(args.out, 'w') as f:
            json.dump({'args': vars(args), 'log': log}, f, indent=1)

    print(f'Final evaluation RMSE ({args.label}): {log[-1]["eval_rmse"]:.4f}')


if __name__ == '__main__':
    main()
