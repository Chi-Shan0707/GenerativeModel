## 2.2 Basic Diffusion Model (SDE Perspective)

These notes refactor the core mathematical ideas of diffusion models into clean Markdown-LaTeX.
The focus is on the SDE formulation and how to simulate it in practice.

---

### 1. Stochastic Process: The Starting Point

A stochastic process is a time-indexed random variable:

$$
(X_t)_{0 \le t \le 1}, \qquad X_t \in \mathbb{R}^d.
$$

Equivalent functional view:

$$
X: [0,1] \to \mathbb{R}^d, \quad t \mapsto X_t.
$$

For each fixed $t$, $X_t$ is random. So if we simulate twice, trajectories can differ.

Freshman step-by-step remark:
1. In ODEs, one initial point gives one fixed path.
2. In SDEs, randomness is injected at each step.
3. Same setup can produce different paths, which is expected.

---

### 2. Brownian Motion (Wiener Process)

A Brownian motion $W=(W_t)_{0 \le t \le 1}$ satisfies:

1. $W_0 = 0$.
2. $t \mapsto W_t$ is continuous.
3. Normal increments:

$$
W_t - W_s \sim \mathcal{N}\bigl(0, (t-s)I_d\bigr), \quad 0 \le s < t.
$$

4. Independent increments:
for $0 \le t_0 < t_1 < \cdots < t_n = 1$, random variables
$W_{t_1}-W_{t_0}, \ldots, W_{t_n}-W_{t_{n-1}}$ are independent.

A simple discrete simulation (step size $h$):

$$
W_{t+h} = W_t + \sqrt{h}\,\varepsilon_t,
\qquad \varepsilon_t \sim \mathcal{N}(0, I_d),
\qquad t=0,h,2h,\ldots,1-h.
$$

Freshman step-by-step remark:
1. Think of Brownian motion as a continuous random walk.
2. Over small time $h$, noise scale is $\sqrt{h}$ (not $h$).
3. Larger time gap means larger variance in increments.

---

### 3. From ODE to SDE

Start from an ODE trajectory:

$$
\frac{d}{dt}X_t = u_t(X_t).
$$

Rewrite in small-step form:

$$
\frac{X_{t+h}-X_t}{h} = u_t(X_t) + R_t(h)
$$

$$
\Longleftrightarrow \quad X_{t+h} = X_t + h\,u_t(X_t) + hR_t(h),
$$

where $R_t(h) \to 0$ as $h \to 0$.

Now add Brownian randomness:

$$
X_{t+h} = X_t + h\,u_t(X_t)
+ \sigma_t\bigl(W_{t+h}-W_t\bigr)
+ hR_t(h).
$$

This corresponds to the symbolic SDE notation:

$$
dX_t = u_t(X_t)\,dt + \sigma_t\,dW_t,
\qquad X_0 = x_0.
$$

Freshman step-by-step remark:
1. ODE part: move along vector field $u_t$.
2. SDE part: add random perturbation from Brownian increments.
3. $\sigma_t$ controls how strong the noise is.
4. If $\sigma_t=0$, SDE reduces to ODE.

---

### 4. Existence and Uniqueness (Informal Statement)

If
- $u: \mathbb{R}^d \times [0,1] \to \mathbb{R}^d$ is continuously differentiable with bounded derivative, and
- $\sigma_t$ is continuous,

then the SDE has a unique solution process $(X_t)_{0\le t\le1}$.

Freshman step-by-step remark:
1. "Well-behaved drift" + "well-behaved noise scale" gives a valid process.
2. "Unique solution" means no ambiguity once randomness is fixed.

---

### 5. Example: Ornstein-Uhlenbeck (OU) Process

Choose

$$
\sigma_t = \sigma \ge 0, \qquad u_t(x) = -\theta x, \quad \theta > 0.
$$

Then:

$$
dX_t = -\theta X_t\,dt + \sigma\,dW_t.
$$

Interpretation:
- Drift term $-\theta X_t$ pulls state toward zero.
- Noise term keeps injecting randomness.

As $t\to\infty$, the stationary distribution is:

$$
X_t \Rightarrow \mathcal{N}\!\left(0, \frac{\sigma^2}{2\theta}\right).
$$

Freshman step-by-step remark:
1. Drift acts like a spring pulling back to center.
2. Noise shakes the state away from center.
3. Balance between pull and shake gives a Gaussian equilibrium.

---

### 6. Euler-Maruyama Simulation for SDEs

For step size $h=1/n$:

$$
X_{t+h} = X_t + h\,u_t(X_t) + \sqrt{h}\,\sigma_t\,\varepsilon_t,
\qquad \varepsilon_t \sim \mathcal{N}(0,I_d).
$$

Freshman step-by-step remark:
1. Compute deterministic update $h\,u_t(X_t)$.
2. Sample Gaussian noise $\varepsilon_t$.
3. Scale noise by $\sqrt{h}\,\sigma_t$ and add it.
4. Repeat until $t=1$.

---

### 7. Diffusion Model as an SDE Generative Model

Parameterize drift with a neural network:

$$
u_\theta: \mathbb{R}^d \times [0,1] \to \mathbb{R}^d,
\qquad (x,t) \mapsto u_t^\theta(x).
$$

Define model:

$$
dX_t = u_t^\theta(X_t)\,dt + \sigma_t\,dW_t,
\qquad X_0 \sim p_{\mathrm{init}}.
$$

Goal:

$$
X_1 \sim p_{\mathrm{data}}.
$$

Freshman step-by-step remark:
1. Start from simple noise distribution $p_{\mathrm{init}}$.
2. Run the learned SDE dynamics from $t=0$ to $t=1$.
3. Hope final samples match real data distribution.

---

### 8. Sampling Algorithm (Euler-Maruyama)

Given network $u_t^\theta$, step count $n$, and diffusion schedule $\sigma_t$:

1. Set $t=0$, $h=1/n$.
2. Sample $X_0 \sim p_{\mathrm{init}}$.
3. For $i=0,1,\dots,n-1$:
   - Sample $\varepsilon_i \sim \mathcal{N}(0,I_d)$.
   - Update

$$
X_{t+h} = X_t + h\,u_t^\theta(X_t) + \sigma_t\sqrt{h}\,\varepsilon_i.
$$

   - Set $t \leftarrow t+h$.
4. Return $X_1$.

Freshman step-by-step remark:
1. It is just "Euler method + Gaussian noise".
2. Smaller $h$ means more steps and usually better approximation.
3. Computational cost grows with number of steps.

---

### 9. Minimal Formula Sheet

$$
\text{SDE:}\quad dX_t = u_t(X_t)dt + \sigma_t dW_t
$$

$$
\text{EM step:}\quad X_{t+h} = X_t + h u_t(X_t) + \sqrt{h}\sigma_t\varepsilon_t
$$

$$
\text{Diffusion model:}\quad dX_t = u_t^\theta(X_t)dt + \sigma_t dW_t,\; X_0\sim p_{\mathrm{init}},\; X_1\sim p_{\mathrm{data}}
$$

Freshman final checklist:
1. Know what Brownian motion is.
2. Understand ODE update vs SDE update.
3. Memorize Euler-Maruyama step.
4. Connect neural drift $u_t^\theta$ to generative modeling.
