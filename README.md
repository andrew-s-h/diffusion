# DDPM from Scratch — CelebA 64×64

A discrete-time denoising diffusion probabilistic model (Ho et al., 2020) implemented from
first principles in PyTorch: epsilon-prediction parameterization, closed-form forward process,
analytic reverse posterior, fixed posterior variance.

**Status.** Diffusion process (`ddpm.py`) and CelebA preprocessing (`load_data.py`) implemented.
The UNet, training loop, and sampling entry point are currently in progress

<!-- TODO: sample grid here once trained -->
<!-- ![Samples](assets/samples_64.png) -->

---

## Formulation

**Forward process.** A fixed Markov chain adds Gaussian noise over $T$ steps according to a
variance schedule $\beta_1, \dots, \beta_T$:

$$q(x_t \mid x_{t-1}) = \mathcal{N}\!\left(x_t;\ \sqrt{1-\beta_t}\,x_{t-1},\ \beta_t \mathbf{I}\right)$$

With $\alpha_t = 1 - \beta_t$ and $\bar\alpha_t = \prod_{s=1}^{t} \alpha_s$, the chain admits a
closed form at arbitrary $t$, which is what makes training tractable — no sequential simulation
is needed to build a training example:

$$q(x_t \mid x_0) = \mathcal{N}\!\left(x_t;\ \sqrt{\bar\alpha_t}\,x_0,\ (1-\bar\alpha_t)\mathbf{I}\right)
\quad\Longleftrightarrow\quad
x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon,\ \ \epsilon \sim \mathcal{N}(0,\mathbf{I})$$

**Reverse posterior.** Conditioned on $x_0$, the reverse step is Gaussian in closed form:

$$q(x_{t-1} \mid x_t, x_0) = \mathcal{N}\!\left(x_{t-1};\ \tilde\mu_t(x_t, x_0),\ \tilde\beta_t \mathbf{I}\right)$$

$$\tilde\mu_t(x_t, x_0) = \frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}\,x_0
+ \frac{\sqrt{\alpha_t}\,(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\,x_t,
\qquad
\tilde\beta_t = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\,\beta_t$$

With the convention $\bar\alpha_0 = 1$, this gives $\tilde\beta_1 = 0$: the final reverse step is
deterministic.

**Training objective.** Substituting the epsilon-parameterization into the variational bound and
dropping the time-dependent weighting gives the simplified loss that's actually being minimizedtim:

$$L_{\text{simple}} = \mathbb{E}_{t \sim \mathcal{U}\{1,T\},\ x_0,\ \epsilon}
\left[\left\lVert \epsilon - \epsilon_\theta\!\left(\sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon,\ t\right) \right\rVert^2\right]$$

A timestep is drawn independently per example rather than once per batch, which reduces the
variance of the gradient estimate.

**Reverse mean.** Rather than evaluating the collapsed form directly, the network's noise
prediction is used to invert the forward marginal,

$$\hat{x}_0 = \frac{1}{\sqrt{\bar\alpha_t}}\,x_t - \sqrt{\frac{1}{\bar\alpha_t} - 1}\ \epsilon_\theta(x_t, t)$$

and $\hat{x}_0$ is substituted into the analytic posterior mean above. Algebraically this is
identical to

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon_\theta(x_t, t)\right)$$

but routing through $\hat{x}_0$ allows clamping it to $[-1, 1]$ — the data range — before forming
the mean. At large $t$ the factor $\sqrt{1/\bar\alpha_t - 1}$ is large, so a small error in
$\epsilon_\theta$ produces an implied $\hat{x}_0$ far outside the data range; clamping prevents
that from propagating down the chain.

**Sampling.** Ancestral sampling from $x_T \sim \mathcal{N}(0, \mathbf{I})$, for $t = T, \dots, 1$:

$$x_{t-1} = \mu_\theta(x_t, t) + \sigma_t z, \qquad \sigma_t^2 = \tilde\beta_t,
\qquad z \sim \mathcal{N}(0,\mathbf{I}) \ \text{for } t>1,\ z = 0 \ \text{at } t=1$$

### Design decisions

| Choice | Taken | Alternative | Rationale |
|---|---|---|---|
| Parameterization | predict $\epsilon$ | predict $x_0$ or $v$ | Empirically better sample quality; the loss reduces to plain MSE against a unit-variance target, so gradients are well-scaled across all $t$ |
| Reverse variance | fixed $\sigma_t^2 = \tilde\beta_t$ | learned $\Sigma_\theta$, or $\sigma_t^2 = \beta_t$ | $\tilde\beta_t$ is the exact posterior variance conditioned on $x_0$; $\beta_t$ is the corresponding bound when the data distribution is treated as maximally diffuse. Learning the variance buys likelihood, not visual quality, at added complexity |
| Schedule | linear, $\beta_1 = 10^{-4} \to \beta_T = 0.02$, $T = 1000$ | cosine | Matches the reference implementation, so results are comparable |

---

## Results

<!-- TODO -->

| Metric | Value |
|---|---|
| FID (50k samples vs. held-out split) | TBD |
| Training loss (final) | TBD |
| Steps / wall-clock | TBD |

Loss curve and sample grids at several checkpoints: assets/

---

## Limitations

- **Sampling is slow.** Ancestral sampling requires $T = 1000$ sequential network evaluations per
  batch. DDIM would cut this by an order of magnitude at some cost in diversity; not implemented.
- **Unconditional only.** No class conditioning and no classifier-free guidance, so there is no
  quality/diversity control knob.
- **64×64 only.** 

---

## Reference

Ho, Jain, Abbeel. *Denoising Diffusion Probabilistic Models.* NeurIPS 2020. arXiv:2006.11239