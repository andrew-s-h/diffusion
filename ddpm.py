import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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

        t = torch.randint(0, self.timesteps, (b, ), device = x0.device, dtype=torch.long)

        eps = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, eps)
        eps_hat = self.model(x_t, t)
        return F.mse_loss(eps_hat, eps)

