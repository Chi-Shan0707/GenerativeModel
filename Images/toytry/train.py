import os
import argparse
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import get_model
from diffusion import DDPM

# Simple training script for the MNIST DDPM demo.
# This script downloads MNIST automatically, trains to predict noise,
# and saves checkpoints to disk.


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')

    transform = transforms.Compose([
        transforms.ToTensor(),
        # map into [-1,1] which DDPM math often assumes
        transforms.Lambda(lambda t: t * 2.0 - 1.0),
    ])

    dataset = datasets.MNIST(root=args.data_dir, train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    model = get_model(device)
    ddpm = DDPM(timesteps=args.timesteps, device=device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 0
    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        for idx, (images, _) in enumerate(loader):
            images = images.to(device)
            loss = ddpm.loss(model, images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if idx % args.log_interval == 0:
                print(f'Epoch {epoch} | Iter {idx}/{len(loader)} | loss {loss.item():.4f}')

        avg_loss = running_loss / len(loader)
        print(f'End epoch {epoch} | avg loss {avg_loss:.4f}')

        # Save checkpoint
        ckpt_path = Path(args.save_dir) / f'ckpt_epoch_{epoch}.pth'
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}, ckpt_path)
        print('Saved', ckpt_path)


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', type=str, default='./data')
    p.add_argument('--save-dir', type=str, default='./checkpoints')
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--timesteps', type=int, default=200)
    p.add_argument('--log-interval', type=int, default=200)
    p.add_argument('--no-cuda', action='store_true')
    return p


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    train(args)
