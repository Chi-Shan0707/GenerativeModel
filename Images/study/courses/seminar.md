## Seminar Notes

Manifold in $\mathbb{R}^d$: $(M, p)$.

A dynamical view of distributions.

Instead of learning $p(x)$ directly,
we use ideas from optimal transport.

From optimal transport, we do not only match two endpoint distributions;
we also obtain a principled way to connect them by a minimal-cost motion.

Let $p_0 := p_{\text{unknown}}$ and $p_1 := p_{\text{gaussian}}$.
Optimal transport gives a coupling (or transport map) that moves mass
from $p_0$ to $p_1$ with minimal expected cost.

This naturally induces a path of intermediate distributions
$(p_t)_{t\in[0,1]}$ (displacement interpolation), interpreted as
the geodesic between $p_0$ and $p_1$ in Wasserstein space.

So we construct a path of distributions:

$$
p_{\text{unknown}} \rightarrow p_{\text{gaussian}}.
$$

In dynamic form, this path is described by a time-dependent velocity field
$v_t(x)$ satisfying the continuity equation:

$$
\partial_t p_t(x) + \nabla \cdot \bigl(p_t(x) v_t(x)\bigr) = 0.
$$

Finally, we model particle dynamics with an SDE:

$$
dX_t = f(X_t, t)\,dt + g(t)\,dW_t.
$$

---

## Time Reversal: Anderson (1982)

**Key insight**: The diffusion process can be reversed to generate data from noise.

### Forward SDE

$$
dX_t = f(X_t, t)\,dt + g(t)\,dW_t, \qquad t \in [0, 1]
$$

- **Drift**: $f(X_t, t)$
- **Diffusion**: $g(t)$
- **Direction**: Data → Noise (forward in time)

### Reverse SDE (Anderson 1982)

$$
dX_t = \bigl[f(X_t, t) - g(t)^2 \nabla \log p_t(X_t)\bigr]\,dt + g(t)\,d\tilde{W}_t, \qquad t \in [1, 0]
$$

- **Drift with correction**: $f(X_t, t) - g(t)^2 \nabla \log p_t(X_t)$
- **Diffusion**: $g(t)$ (same)
- **Score function**: $\nabla \log p_t(X_t) = s_\theta(X_t, t)$
- **Direction**: Noise → Data (backward in time)

### Forward vs Backward Comparison

| | Forward SDE | Reverse SDE |
|---|-------------|-------------|
| **Drift** | $f(X_t, t)$ | $f(X_t, t) - g(t)^2 \nabla \log p_t(X_t)$ |
| **Diffusion** | $g(t)$ | $g(t)$ |
| **Time direction** | $t: 0 \to 1$ | $t: 1 \to 0$ |
| **Noise term** | $dW_t$ | $d\tilde{W}_t$ (reverse Wiener) |
| **Purpose** | Add noise (forward diffusion) | Remove noise (generation) |

**Key insight**: To reverse time, we must add a **score-based drift correction** $-g(t)^2 \nabla \log p_t(X_t)$ that "pushes back" against the diffusion.

---

## Numerical Methods for SDEs

To simulate SDEs numerically, we discretize time with step size $\Delta t$.
Let $t_n = n\Delta t$ and $X_n \approx X_{t_n}$.

### Euler-Maruyama Method

For the SDE $dX_t = a(X_t, t)dt + b(X_t, t)dW_t$:

$$
X_{n+1} = X_n + a(X_n, t_n)\Delta t + b(X_n, t_n)\Delta W_n
$$

where $\Delta W_n \sim \mathcal{N}(0, \Delta t)$ (Wiener increment).

**Convergence**:
- Strong order: 0.5 ($\mathbb{E}[|X_T - \hat{X}_T|] \leq C \sqrt{\Delta t}$)
- Weak order: 1.0 ($|\mathbb{E}[f(X_T)] - \mathbb{E}[f(\hat{X}_T)]| \leq C \Delta t$)

### Milstein Scheme (Higher-Order Stochastic RK)

$$
X_{n+1} = X_n + a(X_n, t_n)\Delta t + b(X_n, t_n)\Delta W_n + \frac{1}{2}b(X_n, t_n)\frac{\partial b}{\partial x}(X_n, t_n)(\Delta W_n^2 - \Delta t)
$$

**Additional term**: The **diffusion derivative** $\frac{\partial b}{\partial x}$ captures second-order Itô correction.

**Convergence**:
- Strong order: 1.0 ($\mathbb{E}[|X_T - \hat{X}_T|] \leq C \Delta t$)
- Weak order: 1.0

### Comparison

| Method | Strong Order | Weak Order | Complexity | When to Use |
|--------|-------------|------------|------------|-------------|
| **Euler-Maruyama** | 0.5 | 1.0 | Low | Simple problems, large $\Delta t$ |
| **Milstein** | 1.0 | 1.0 | Medium (needs $\partial b/\partial x$) | Higher accuracy needed |

**Tradeoff**: Milstein achieves $2\times$ better strong convergence but requires computing diffusion derivatives.

---

## Diffusion Models = Learn Score Function

The core idea of modern diffusion models:

$$
\text{Train neural network } s_\theta(x, t) \approx \nabla_x \log p_t(x)
$$

**Why learn the score?**
- Score $\nabla \log p_t(x)$ points toward high-probability regions
- Reverse SDE needs score to generate data: $dX_t = [f - g^2 s_\theta(X_t, t)]dt + g d\tilde{W}_t$
- Score matching provides tractable training objective

### Three Parameterizations

Different ways to parameterize the learning objective:

| Type | What to Predict | Training Target | Relationship to Score |
|------|-----------------|-----------------|----------------------|
| **Noise Prediction** ($\epsilon$-prediction) | Noise $\epsilon_t$ | $\| \epsilon_t - \epsilon_\theta(x_t, t) \|^2$ | $\epsilon_\theta = -\sigma_t \cdot \nabla \log p_t$ |
| **Data Prediction** ($x_0$-prediction) | Original data $x_0$ | $\| x_0 - x_{\theta}(x_t, t) \|^2$ | $x_\theta$ directly estimates clean data |
| **Speed / Velocity** ($v$-prediction) | Velocity field $v_t$ | $\| v_t - v_\theta(x_t, t) \|^2$ | $v_\theta = \frac{dx_t}{dt}$ (straight paths) |

**Mathematical equivalence**: All three formulations are equivalent with proper scaling:
$$
\epsilon_t = \frac{x_t - \sqrt{\bar{\alpha}_t} x_0}{\sqrt{1-\bar{\alpha}_t}}, \quad v_t = \alpha_t \epsilon_t - \sigma_t \nabla \log p_t
$$

### From SDE to ODE: Acceleration

**Problem**: Reverse SDE requires many steps (1000+ in DDPM) due to stochastic noise.

**Solution**: Use Probability Flow ODE (Song et al., 2021)

$$
dX_t = \bigl[f(X_t, t) - \frac{1}{2}g(t)^2 \nabla \log p_t(X_t)\bigr]\,dt
$$

**Key difference from SDE**: No noise term $dW_t$ → deterministic sampling!

| SDE Sampling | ODE Sampling |
|--------------|--------------|
| Stochastic ($dW_t$ present) | Deterministic (no $dW_t$) |
| Slow (1000 steps) | Fast (10-50 steps) |
| Same marginal distribution $p_t$ | Same marginal distribution $p_t$ |
| DDPM, Score-SDE | DDIM, DPM-Solver, Flow Matching, Rectified Flow |

### Fast Sampling Methods

| Method | Steps | Type | Description |
|--------|-------|------|-------------|
| **DDPM** | 1000 | SDE | Original stochastic sampling |
| **DDIM** | 20-50 | ODE | First deterministic ODE sampling |
| **DPM-Solver** | 10-25 | ODE | High-order ODE solver |

**DDIM (Denoising Diffusion Implicit Models)**: First to use deterministic ODE-style sampling for 10-50× acceleration.

**DPM-Solver** (Lu et al., 2022): Dedicated high-order ODE solver achieving 10-25 step sampling with quality comparable to 1000-step DDPM.

### DDPM Essentials (Denoising Diffusion Probabilistic Models)

If "DPPM" means DDPM, the core formulation is:

Forward noising Markov chain:

$$
q(x_t\mid x_{t-1}) = \mathcal N\!\left(\sqrt{1-\beta_t}\,x_{t-1},\,\beta_t I\right),
\quad t=1,\dots,T.
$$

Define $\alpha_t=1-\beta_t$ and $\bar\alpha_t=\prod_{s=1}^t\alpha_s$.
Then the closed form from clean data to step $t$ is:

$$
q(x_t\mid x_0)=\mathcal N\!\left(\sqrt{\bar\alpha_t}\,x_0,\,(1-\bar\alpha_t)I\right),
$$

equivalently

$$
x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon,
\qquad \epsilon\sim\mathcal N(0,I).
$$

Reverse denoising model:

$$
p_\theta(x_{t-1}\mid x_t)=\mathcal N\!\left(\mu_\theta(x_t,t),\,\Sigma_\theta(x_t,t)\right).
$$

Practical training objective (noise prediction / $L_{\text{simple}}$):

$$
\mathcal L_{\text{simple}}
=\mathbb E_{t,x_0,\epsilon}\left[\left\|\epsilon-\epsilon_\theta(x_t,t)\right\|_2^2\right].
$$

Connection to score learning:

$$
\nabla_{x_t}\log q(x_t\mid x_0)
=-\frac{1}{\sqrt{1-\bar\alpha_t}}\,\epsilon,
$$

so predicting $\epsilon$ is equivalent (up to scaling) to predicting score.

Improved DDPM (Nichol & Dhariwal, 2021):
- Learn reverse variances for better likelihood-quality tradeoff.
- Better noise schedules (e.g., cosine) improve training and sampling.
- Much fewer sampling steps are possible with small quality drop.

**Modern ODE methods**:
- **Flow Matching** (Lipman et al., 2023): Learn straight-line trajectories
- **Rectified Flow** (Liu et al., 2022): Flatten trajectories via iterative reflow

### Rectified Flow (More Details)

Rectified Flow learns an ODE transport that prefers straight trajectories between two distributions.

Let $X_0 \sim \pi_0$ (usually noise) and $X_1 \sim \pi_1$ (data).
Define linear interpolation:

$$
X_t = (1-t)X_0 + tX_1, \qquad t\in[0,1].
$$

The corresponding target velocity along this path is:

$$
\dot X_t = X_1 - X_0.
$$

Train a neural velocity field $v_\theta(x,t)$ by least squares:

$$
\min_\theta \; \mathbb{E}_{t\sim \mathcal U[0,1],\,(X_0,X_1)\sim \gamma}
\left[\left\|v_\theta(X_t,t) - (X_1-X_0)\right\|^2\right],
$$

where $\gamma$ is a coupling between $\pi_0$ and $\pi_1$.

After training, sample by solving the ODE:

$$
\frac{dZ_t}{dt}=v_\theta(Z_t,t), \qquad Z_0\sim \pi_0.
$$

A one-step Euler approximation is:

$$
Z_1 \approx Z_0 + v_\theta(Z_0,0),
$$

which works well when trajectories are close to straight lines.

Why it is fast:
- No stochastic term during sampling (ODE, not reverse SDE).
- Straighter trajectories reduce discretization error.
- Reflow/rectification can iteratively straighten trajectories further.

---

## MAE vs Diffusion: Kaiming He's Alternative

**MAE (Masked Autoencoders, Kaiming He et al., 2021)**: A different paradigm—no gradual noise, just masking.

| Aspect | MAE (Kaiming He) | Diffusion |
|--------|------------------|-----------|
| **Corruption** | Direct mask (75% patches removed) | Gradual noise addition (1000 steps) |
| **Process** | One-step masking | Multi-step diffusion |
| **Training** | Fast: reconstruct from visible patches | Slow: predict noise at each timestep |
| **Goal** | Representation learning | Generation |
| **Architecture** | Asymmetric encoder-decoder | Variable |

**Kaiming He's insight**: Why add noise gradually when you can simply mask and reconstruct?
- Masking is simpler and faster
- High mask ratio (75%) forces learning meaningful representations
- Non-symmetric: encoder sees only visible patches (25% compute)

$$
\text{MAE: } x \xrightarrow{\text{mask 75%}} \tilde{x} \xrightarrow{\text{Encoder}} z \xrightarrow{\text{Decoder}} \hat{x} \approx x
$$

$$
\text{Diffusion: } x \xrightarrow{1000 \text{ steps}} x_T \sim \mathcal{N}(0,I) \xrightarrow{\text{reverse SDE}} \hat{x} \approx x
$$

**Key difference**: MAE learns representations (for downstream tasks); diffusion learns to generate (for creating new data).

---

## Diffusion Transformer (DiT)

Architecture evolution for diffusion models:

```
DDPM (UNet backbone) → DiT (Transformer backbone)
     ↓                          ↓
   Convolution              Self-Attention
   Spatial inductive bias    Global context
```

**DiT** (Peebles & Xie, 2023): Replaces UNet with Vision Transformer
- Uses standard Transformer blocks (self-attention + MLP)
- Patchifies images like ViT
- Scale-friendly: better scaling than UNet

**Combined with v-parameterization**: DiT + speed field achieves SOTA generation quality (e.g., Stable Diffusion 3, FLUX).

---

## Summary: Evolution of Diffusion

```
DDPM (2020)
    │
    ├─→ Learn score function via noise prediction
    │
    ├─→ DDIM (2021): ODE for acceleration
    │
    ├─→ Probability Flow ODE (Song & Yang, 2021)
    │   └─→ Flow Matching (2023)
    │   └─→ Rectified Flow (2022)
    │
    ├─→ v-parameterization / Speed field (Progressive Distillation, 2022)
    │
    └─→ DiT (2023): Transformer architecture
        └─→ DiT + v-pred (Modern SOTA: SD3, FLUX)
```

**Core insight**: Diffusion models learn to reverse the diffusion process. The reverse SDE needs the score function, which can be learned via noise prediction, data prediction, or velocity prediction. For faster sampling, replace stochastic SDE with deterministic ODE.
