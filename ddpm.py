import copy
import math
import os 
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

class Gaussian_Diffusion(nn.Module):
    """Discrete-time denoising diffusion probability model: schedule,
    forward/reverse Gaussians,loss, and sampler

    Schedule-derived constants are precomuted once at construction using float64 accuracy,
    and are stored as float32 buffers so .to(device) moves them w/the module and exposes:

        q_sample:       draw x_t ~ q(x_t | x_0)
        q_posterior:    mean/variance of q(x_{t-1} | x_t, x_0)
        p_mean_variance: mean/variance of p_theta(x_{t-1} | x_t)
        p_sample:       one step backwards x_t -> x_t-1
        sample:         full-reverse from x_T -> x_0
        loss:           L_simple for a batch of clean images

    Args: 
        model (nn.Module): eps_theta network; model(x_t, t) -> noise prediction
            same shape as x_t, where t is a (B, ) long tensor of timestep indices
            timesteps (int): number of diffusion steps T. Default value 1000
            beta_start (float): beta_1 of the linear schedule. Default 1e-4
            beta_end (float): beta_T of the linear schedule. Default 0.02. 

    
    """
    def __init__(self, model: nn.Module, timesteps: int=1000, 
                 beta_start: float = 1e-4, beta_end: float = 0.02):
        super().__init__()
        self.model = model
        self.timesteps = timesteps


        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)

        alphas = 1.0 - betas
        alphas_bar= torch.cumprod(alphas, dim=0)

        alphas_bar_prev = torch.cat([torch.ones(1, dtype=torch.float64), alphas_bar[:-1]])

        # B_tilde
        posterior_variance = betas * ((1.0-alphas_bar_prev) / (1.0 - alphas_bar))

        # registers as float32 buffers
        def buf(name, tensor):
            self.register_buffer(name, tensor.float())

        buf("betas", betas)
        # this is for the forward marginal q(x_t|x_0)
        # q_sample: x_t = sqrt(alphas_bar_t) * x_0 + sqrt(1-alpha_bar_t) * eps
        buf("sqrt_alphas_bar", torch.sqrt(alphas_bar))
        buf("sqrt_one_minus_alphas_bar", torch.sqrt(1.0 - alphas_bar))

        # for inverting the marginal 
        # x_hat_0 = (1/sqrt(alpha_bar_t)) * x_t  - sqrt((1/alpha_bar_t) - 1)* noise_estimate
        buf("sqrt_recip_alphas_bar", torch.sqrt(1.0/alphas_bar))
        buf("sqrt_recip_sub1_alphas_bar", torch.sqrt(1.0/alphas_bar - 1))

        # Analytic posterior
        buf("posterior_variance", posterior_variance)
        buf("posterior_mean_coef1", betas * torch.sqrt(alphas_bar_prev) / (1.0 - alphas_bar))
        buf("posterior_mean_coef2", (1.0 - alphas_bar_prev) * torch.sqrt(alphas) / (1.0 - alphas_bar))

    def _extract(self, schedule: torch.Tensor, t: torch.Tensor, ndim:int) -> torch.Tensor:
        """Gathers per-example schedule values and reshapes for broadcasting

        Can think of it as taking schedule tensor, taking indices specified in t,
        and returning values from scheudle. 
        
        Args: 
            schedule (torch.Tensor): size (T, ) buffer of per-timestep scheduler values
            t (torch.Tensor): (B, ) long tensor of timestep indices
            ndim (int): rank of the tensor the result will multiply 

    
        Returns:
            torch.Tensor: (B, 1, ..., 1) values
        """
        return schedule[t].view(-1, *([1] * (ndim - 1)))

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor,  
                 eps: torch.Tensor) -> torch.Tensor:
        """Draws x_t ~ q(x_t | x_0)
        
        Implements the following:
            x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon

            Args:
                x_0 (torch.Tensor): (B, C, H, W) clean images in [-1, 1].
                t (torch.Tensor): (B,) long timestep indices! [0, T)
                eps (torch.Tensor): (B, C, H, W) standard normal noise
    
            Returns:
                torch.Tensor: (B, C, H, W) noised images x_t
        """    
        return (self._extract(self.sqrt_alphas_bar, t, x_0.ndim) * x_0 +
                self._extract(self.sqrt_one_minus_alphas_bar, t, x_0.ndim) * eps)

    def predict_x0_from_eps(self, x_t: torch.Tensor, t: torch.Tensor,
                                eps_hat: torch.Tensor) -> torch.Tensor:
        """ Inverts the forward marginal; to compute x_0_hat using an epsilon prediction
        
        Computes x_0 = sqrt(1/alpha_bar) * x_t - sqrt(1/alpha_bar - 1) * eps_hat
        
        Args:
            x_t (torch.Tensor): (B, C, H, W) noised images
            t (torch.tensor): (B, ) long timestep indices
            eps_hat (torch.Tensor): (B, C, H, W) the predicted noise

        Returns:
            torch.Tensor: (B, C, H, W) 'implied' clean images x_0_hat
        """
        return (self._extract(self.sqrt_recip_alphas_bar, t, x_t.ndim) * x_t 
            - self._extract(self.sqrt_recip_sub1_alphas_bar, t, x_t.ndim) * eps_hat)

    def q_posterior(self, x_0: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor):
        """
        Computes the analytic posterior q(x_{t-1} | x_t, x_0)

        Pure schedule arithmetic; no parameters and no network:
            mean = posterior_mean_coef1[t] * x_0 +posterior_mean_coef2[t] * x_t
            variance = b_tilde_t

        The Bayes derivation is exact for true x_0. nit the sampler has no true x_0.
        Thus, we pass in x_0_hat from predict_x0_from_eps

        Args:
            x_0 (torch.tensor): (B, C, H, W) clean images. x_0_hat in practice
            x_t (torch.tensor): (B, C, H, W) noisy images
            t (torch.Tensor): (B, ) timestep indices

        Returns:
            tuple[torch.Tensor, torch.Tensor]: tuple containing posterior mean 
            (B, C, H, W) and variance (B, 1, 1, 1)
        """
        mean = (self._extract(self.posterior_mean_coef1, t, x_t.ndim) * x_0
                + self._extract(self.posterior_mean_coef2, t, x_t.ndim) * x_t)

        variance = self._extract(self.posterior_variance, t, x_t.ndim)

        return mean, variance

    def p_mean_variance(self, x_t: torch.Tensor, t: torch.Tensor, 
                        clip_x0: bool = True):
        """
        Computes mean/variance of p_theta(x_{t-1} | x_t)
        
        Called once per step by p_sample, which draws x_{t-1} ~ N(mean, variance)

        Args:
            x_t (torch.Tensor): (B, C, H, W) current reverse-chain state
            t (torch.Tensor): (B,) time step indices 
            clip_x0 (bool): clamp x_0_hat to [-1, 1] before the posterior.
                Data lives in [-1, 1]. Default True
        
        Returns:
            tuple[torch.Tensor, torch.Tensor]: reverse mean (B, C, H, W) & variance (B, 1, 1, 1)
        """
        eps_hat = self.model(x_t, t)
        # x_0 prediction
        x_0_hat = self.predict_x0_from_eps(x_t, t, eps_hat)
        if clip_x0:
            x_0_hat = x_0_hat.clamp(-1.0, 1.0)
        # feed in x_0 to get our q_posterior 
        return self.q_posterior(x_0_hat, x_t, t)
        
    @torch.no_grad()
    def p_sample(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Runs one ancestral step: x_t -> x_{t-1}

        Draws x_{t-1} = mu_theta +sqrt(b_tilde_t) * z with z ~ N(0, I), 
        except at t=0 where no noise is added. The final step returns 
        just the mean

        Args:
            x_t (torch.Tensor): (B, C, H, W) current state.
            t (torch.Tensor): (B, ) timestep indices
        
        Returns:
            torch.Tensor: (B, C, H, W) the sampled x_{t-1} 
        """
        mean, variance = self.p_mean_variance(x_t, t)
        z = torch.randn_like(x_t)
        # check for non-zero indices 
        nonzero = (t>0).float().view(-1, *([1] * (x_t.ndim - 1)))
        return mean + nonzero * torch.sqrt(variance) * z

    @torch.no_grad()
    def sample(self, shape: tuple) -> torch.Tensor:
        """Generates images by running the full reverse chain x_T -> x_0

        Starts from pure noise x_t ~ N(0, I) and applies p_sample T times

        Args: 
            shape (tuple): output shape, e.g. (n, 3, 64, 64)
        
        Returns: 
            torch.Tensor: (shape) generated images, approximately in [-1, 1]
        """
        # check where registered buffer lives...
        device = self.betas.device
        x = torch.randn(shape, device=device)
        for i in reversed(range(self.timesteps)):
            t = torch.full((shape[0], ), i, device = device, dtype = torch.long)
            x= self.p_sample(x, t)
        return x

    def loss(self, x_0: torch.Tensor) -> torch.Tensor:
        """Computes loss for one batch of clean images. 

        Samples t ~ Uniform{0...T-1} & eps ~ N(0, I) per example, forms
        x_t with q_sample, & returns mean squared error between true and predicted
        noise
    
        Args: 
            x0 (torch.Tensor): (B, C, H, W) clean images in [-1, 1]

        Returns: 
            torch.Tensor: scalar loss
        """

        b = x_0.shape[0]

        t = torch.randint(0, self.timesteps, (b, ), device = x_0.device, dtype=torch.long)

        eps = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, eps)
        eps_hat = self.model(x_t, t)
        return F.mse_loss(eps_hat, eps)

# ======================================
# eps_theta network: UNet used for noise prediction with time conditioning
#
# =======================================

class Sinusoidal_Time_Embedding(nn.Module):
    """
    Maps timesteps to fixed featured vetors using a sinusoidal time embeding

    Indexed by integer diffusion step t with h = dim/2

        omega_i = exp(-ln(10000) * i / (h - 1)), i = 0...h-1
        emb(t) = [sin(omega_0)]
    """

    def __init__(self, dim:int):
        super().__init__()
        # sin/cos pairs so need following check...
        assert dim % 2 == 0 

        half = dim//2 
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32) 
                          / (half - 1))
        self.register_buffer("freqs", freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t (torch.Tensor): (B,) long timestep indices
        
        Returns:
            torch.Tensor: (B, dim) float32 embeddings
        """
        # (B, 1) * (1, half) -> (B, half)
        angles = t.float()[:, None] * self.freqs[None, :]
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim =-1)

class Residual_block(nn.Module):
    """
    Group norm -> SiLU -> Conv

    Args:
        in_ch (int): input channels
        out_ch (int): output channels
        time_dim (int): width of shared time embedding vector
        dropout (float): dropout prob before second conv. Default 0.1
    """
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, dropout: float = 0.1):
        super().__init__()

        self.norm1 = nn.GroupNorm(num_groups=32,num_channels=in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

        self.time_proj = nn.Linear(in_features=time_dim, out_features=out_ch)

        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): (B, in_ch, H, W)
            t_emb (torch.Tensor): (B, time_dim) shared time embedding

        Returns:
            torch.Tensor: (B, out_ch, H, W)
        """
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)

class Self_Attention_Block(nn.Module):
    """ UNet drops one of these after every ResBlock sitting at a resolution listed in attn_resolutions 
    -- just (16,) by default -- plus one between the two middle ResBlocks, which is always there regardless
    of that setting. Six in the default network: two in the encoder, three in the decoder, one in the middle at 8x8.

    Args:
        ch (int): channels
        num_heads (int): attention heads; Default 4. Channels must be divisible by num_heads
    """
    def __init__(self, ch:int, num_heads: int=4):
        super().__init__()
        assert ch % num_heads == 0
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(32, ch)

        # 1x1 filter at each position independently: linear in the channel
        # vector, no spatial aggregation. c -> 3c stacks q,k,v into one matmul.
        self.qkv = nn.Conv2d(ch, 3 * ch, kernel_size=1)

        # W_o, each head wrote into its own slice of channels 
        self.proj = nn.Conv2d(ch, ch, kernel_size=1)

        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x:torch.tensor) -> torch.Tensor: 
        """
        Args:
            x (torch.Tensor): (B, ch, H, W)

        Returns: 
            torch.Tensor: (B, ch, H, W)
        """
        b, c, h_, w_ = x.shape
        n = h_ * w_
        head = c // self.num_heads

        qkv = self.qkv(self.norm(x)) # (B,  3c, H, W)

        qkv = qkv.reshape(b, 3, self.num_heads, head, n) # (B, 3, num_heads, head, n)
        qkv = qkv.permute(1, 0, 2, 4, 3) # (3, B, num_heads, n= hw, head)
        q, k, v = qkv[0], qkv[1], qkv[2] # each (B, heads, HW, head)

        # returns output. The weights softmas(q * k^T / sqrt(head)) is computed internally
        # this is a fused kernel, (B,heads,HW,HW) score matrix is never materialized
 
        out = F.scaled_dot_product_attention(q, k, v) # (B, heads, HW, head)
        out = out.permute(0, 1, 3, 2).reshape(b, c, h_, w_)
        return x + self.proj(out)

class Downsample(nn.Module):
        """Halves H and W with a stride-2 3x3 conv (learned reduction, Ho et al.)
        """
        def __init__(self, ch: int):
            super().__init__()
            self.conv = nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.conv(x)

class Upsample(nn.Module):
        """Doubles H and W: nearest-neighbor x2 then a 3x3 conv"""
        def __init__(self, ch: int):
            super().__init__()
            self.conv = nn.Conv2d(ch, ch, kernel_size=3, padding=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


class UNet(nn.Module):
    """
    Noise-prediction network eps_theta(x_t, t)
    
    Encoder halves the resultion at each level, and a middle block sits at the coarsest resolution,
    and a decoder is used to mirror the encoder. The decoder runs num_res_block + 1 ResBlocks per level 


    Args:
        in_ch (int): image channels. Default 3
        base_ch (int): channels at full resolution; multiple of 32 (GroupNorm).
            Default 128
        ch_mults (tuple): per-level multipliers of base_ch; each entry after the first adds one 2x downsample.
            Default (1, 2, 2, 2)
        num_res_block (int): ResBlocks per encoder level. Default 2
        attn_resolutions (tuple): spatial sizes where attention is added for resblocks. Default (16, )
        img_size (int): input resolution, used only to know which level is at
        

    """
    def __init__(self, in_ch: int = 3, base_ch: int = 128,
                 ch_mults: tuple = (1,2,2,2), num_res_blocks: int = 2,
                 attn_resolutions: tuple = (16,), img_size: int = 64, 
                 dropout: float = 0.1):
        super().__init__()

        # number of stages for the encoder/decoder
        self.num_levels = len(ch_mults)
        time_dim = base_ch * 4

        def make_res(cin, cout, res):
            block = Residual_block(cin, cout, time_dim, dropout)
            attn = Self_Attention_Block(ch=cout) if res in attn_resolutions else nn.Identity()
            return nn.ModuleList([block, attn])

        self.time_mlp = nn.Sequential(
            Sinusoidal_Time_Embedding(base_ch),
            nn.Linear(base_ch, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

        # 'lift RGB' into feature channels 
        self.stem = nn.Conv2d(in_ch, base_ch, kernel_size=3, padding=1)

        # ----------- Encoder -------------
        # the encoder's ordered container is self.down
        self.down = nn.ModuleList()
        # skip_chs keeps track of every activation the encoder pushes
        # so the decoder can be built w matchign widths
        skip_chs = [base_ch]
        ch = base_ch
        res = img_size

        for level, mult in enumerate(ch_mults):
            out_ch = base_ch * mult
            for _ in range(num_res_blocks):
                self.down.append(make_res(ch, out_ch, res))
                ch = out_ch
                skip_chs.append(ch)
            # for default case true for levels 0, 1, 2
            if level != self.num_levels - 1:
                self.down.append(Downsample(ch))
                skip_chs.append(ch)
                # bookeeping res, actual change in stride 2 downsample
                # H and W must be even 
                res //= 2

        # ----------- Bottom U-Net -----------

        self.mid_block1 = Residual_block(ch, ch, time_dim, dropout)
        self.mid_attn = Self_Attention_Block(ch)
        self.mid_block2 = Residual_block(ch, ch, time_dim, dropout)

        # ----------- Decoder -----------
        self.up = nn.ModuleList()
        for level, mult in reversed(list(enumerate(ch_mults))):
            out_ch = base_ch * mult

            # decoder uses 3 resblocks per level 
            for _ in range(num_res_blocks + 1):
                self.up.append(make_res(ch + skip_chs.pop(), out_ch, res))
                ch = out_ch

            if level != 0:
                self.up.append(Upsample(ch))
                res *= 2

        # last layer/op set 
        # this will return eps_hat
        self.out_norm = nn.GroupNorm(32, ch)
        self.out_conv = nn.Conv2d(ch, out_channels=in_ch, kernel_size=3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predicts the noised mixed into x_t

        Args:
            x (torch.Tensor): (B, in_ch, H, W) noised images x_t; H, W both divisible
            by 2^(num_levels - 1)

            t (torch.Tensor): (B, ) long timstep indices

        Returns:
            torch.Tensor: (N, in_ch, H, W) predicted noise eps_hat
        """
        t_emb = self.time_mlp(t)
        h = self.stem(x)
        skips = [h]

        for entry in self.down:
            if isinstance(entry, Downsample):
                h = entry(h)
            else:
                block, attn = entry
                # block is resblock
                # resblock forward(self, x, t_emb)
                h = attn(block(h, t_emb))
            skips.append(h)

        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        for entry in self.up:
            if isinstance(entry, Upsample):
                h = entry(h)
            else:
                block, attn = entry
                h = torch.cat([h, skips.pop()], dim=1)
                h = attn(block(h, t_emb))

        assert not skips, "every pushed skip should be popped"
        
        return self.out_conv(F.silu(self.out_norm(h)))


class EMA:
    """Exponential movin average of model parameters
    
    DDPM Samples drawn from EMA wights; Ho et al. use decay 0.9999.

    Averaging smooths the noise that Adam's per-step updates put into
    the weight, which will provide better sample quality.
        shadow <- decay * shadow + (1-decay) * param
    
    Args:
        model (nn.Module): the live model to track
        decay (float): EMA decay. Default 0.9999 
    
    """
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            # takes a step towards live weights
            # s = s + (1-decay) * (p-s)
            s.lerp_(p, 1.0 - self.decay)
