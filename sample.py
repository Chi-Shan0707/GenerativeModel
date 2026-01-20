import argparse
import torch
from pathlib import Path
from diffusion import DDPM
from model import get_model
from utils import save_image_grid


def sample(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')

    model = get_model(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    ddpm = DDPM(timesteps=args.timesteps, device=device)

    samples = ddpm.p_sample_loop(model, (args.num_samples, 1, 28, 28))
    # Save images
    out_path = Path(args.out_dir) / 'samples.png'
    save_image_grid(samples, str(out_path), nrow=args.nrow)
    print('Saved samples to', out_path)


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=str, default='./checkpoints/ckpt_epoch_0.pth')
    p.add_argument('--out-dir', type=str, default='./samples')
    p.add_argument('--num-samples', type=int, default=64)
    p.add_argument('--nrow', type=int, default=8)
    p.add_argument('--timesteps', type=int, default=200)
    p.add_argument('--no-cuda', action='store_true')
    return p


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    torch.manual_seed(0)
    sample(args)
