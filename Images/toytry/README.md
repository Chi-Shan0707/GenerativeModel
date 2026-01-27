# MNIST Diffusion (DDPM) — Minimal Educational Demo

This repository contains a minimal, beginner-friendly implementation of
the DDPM diffusion model trained on MNIST (28x28 grayscale).

What this project demonstrates
- Forward diffusion: gradually adding Gaussian noise to images (q(x_t | x_0)).
- Reverse denoising: learned denoising network predicting noise ε_θ(x_t, t).
- Training: learning to predict the added noise (simple MSE loss).
- Sampling: starting from Gaussian noise and iteratively denoising to generate images.

Files
- `model.py`: small UNet-like network that predicts noise ε_θ(x_t, t). Includes a simple timestep embedding.
- `diffusion.py`: DDPM math: beta schedule, q_sample, loss, and reverse sampling p(x_{t-1}|x_t).
- `utils.py`: helper functions (sinusoidal timestep embedding, image saving).
- `train.py`: training script that downloads MNIST, trains the model, and saves checkpoints.
- `sample.py`: sampling script to load a checkpoint and generate images.
- `requirements.txt`: minimal dependencies (`torch`, `torchvision`).

Intuition (short)
Diffusion models work by corrupting data with noise through many small steps, then
learning a reverse process that gradually removes noise. Training is done by
sampling a random timestep t, corrupting the image to obtain x_t, and training
a neural network to predict the noise that was added. At sampling time we start
from pure Gaussian noise and apply the learned denoising steps backwards.

Quickstart
1. Install dependencies (preferably in a virtual environment):

```bash
pip install -r requirements.txt
```

2. Train (small default settings):

```bash
python train.py --epochs 3 --batch-size 128 --timesteps 200
```

Checkpoints will be saved in `./checkpoints` by default.

3. Sample images from a checkpoint:

```bash
python sample.py --checkpoint ./checkpoints/ckpt_epoch_2.pth --num-samples 64
```

Recommended hyperparameters for beginners
- `timesteps=100..200`: fewer timesteps is faster for learning and fine for a demo.
- `batch-size=64..256`: tune to fit your GPU/CPU memory.
- `epochs=3..10`: more epochs improve quality gradually.
- `lr=2e-4`: reasonable default for Adam.

Notes on DDPM relation
- This demo follows the Ho et al. DDPM formulation: a fixed forward noise schedule
  and a model that predicts noise ε_θ(x_t, t). The training loss is MSE between
  predicted and true noise. The sampling uses the closed-form posterior mean and
  variance to sample x_{t-1} from x_t.
- The model here is intentionally small and not optimized for sample fidelity —
  the goal is clarity and educational value.

Further reading
- The original DDPM paper: "Denoising Diffusion Probabilistic Models" (Ho et al.)

If you want, I can:
- Run a short training job locally and save an example checkpoint + samples.
- Reduce model size further or add more comments to any file.

## Debug Notes

- What was wrong:
  - The initial `SmallUNet` concatenated the timestep embedding into the
    bottleneck feature map, which doubled the number of channels unexpectedly.
    That caused `ConvTranspose2d` in the decoder to receive inputs with the
    wrong channel size and raised runtime shape errors.
  - The `_extract` helper previously attempted to `gather` and `view` into the
    full broadcast shape which could fail when shapes didn't align exactly.

- How it was fixed:
  - `model.py`: project the timestep embedding to the same number of
    bottleneck channels and *add* it to the bottleneck features instead of
    concatenating. This preserves channel counts and keeps skip-connections
    aligned with decoder expectations.
  - `diffusion.py`: `_extract` now indexes the 1-D schedule with the
    batch timesteps and reshapes/`expand`s to the requested broadcast shape.
    This guarantees tensors used in formulas have identical shapes for safe
    elementwise math.
  - Added inline comments and assertions in `model.py` to document tensor
    shapes at key points and prevent silent mismatches.

- Common pitfalls in UNet-based diffusion models:
  - Be explicit about channel counts when concatenating skip connections.
  - Prefer adding projected timestep embeddings (same channel count) over
    concatenation unless you also adjust later layers to accept increased
    channels.
  - Always check that model output shape matches the training target (noise)
    before computing the loss.

