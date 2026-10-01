"""Tests for ddpm.py
 
Run from the project root:
    pytest -m "not slow"     fast tests only
    pytest                   everything, including the overfit test
"""
 
import pytest
import torch
import torch.nn as nn
 
from ddpm import Gaussian_Diffusion, UNet
 
# ================================================================ helpers
 
 
class DummyNet(nn.Module):
    """Stand-in for eps_theta that always predicts zero noise"""
 
    def __init__(self):
        super().__init__()
        # one parameter so the module has something for .parameters() to return
        self.scale = nn.Parameter(torch.zeros(1))
 
    def forward(self, x_t, t):
        assert t.shape == (x_t.shape[0],) and t.dtype == torch.long
        return torch.zeros_like(x_t) + self.scale
 
 
def wake_up(model):
    """Replace the zero-initialized weights so the output depends on the input"""
    with torch.no_grad():
        for p in model.parameters():
            if torch.all(p == 0):
                p.normal_(0.0, 0.1)
    return model

# ================================================================ fixtures

@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)

@pytest.fixture
def diff():
    """Default T=1000 linear schedule. Uses beta_start = 1e^-4 & beta_end = 0.02"""
    return Gaussian_Diffusion(DummyNet(), timesteps=1000)

@pytest.fixture
def x_0():
    """A small batch of fake images in [-1,1]"""
    return torch.rand(4, 3, 8, 8) * 2.0 - 1.0

@pytest.fixture
def tiny_unet():
    """"16 x 16 -> 8x8 Unet with attention at 8x8 resolution"""
    return UNet(base_ch=32, ch_mults=(1, 2), attn_resolutions=(8,), img_size=16)

def test_roundtrip_revoverd_x0(diff, x_0):
    """It should return exactly x_0 when true eps is passed to predict_x0_from_eps"""
    t = torch.randint(0, diff.timesteps, (x_0.shape[0], ))
    t[0], t[-1] = 0, diff.timesteps - 1
    eps = torch.randn_like(x_0)
    x_t = diff.q_sample(x_0, t, eps)
    x_0_hat = diff.predict_x0_from_eps(x_t, t, eps)

    assert (x_0_hat - x_0).abs().max() < 1e-3


def test_schedule_buffers(diff):
    """It should store float32 (T,) buffers matching a float64 recomputation, with a_bar strictly decreasing to ~0"""
    T = diff.timesteps
    betas = torch.linspace(1e-4, 0.02, T, dtype=torch.float64)
    alphas_bar = torch.cumprod(1.0 - betas, dim=0)

    for name, buf in diff.named_buffers():
        assert buf.dtype == torch.float32, name
        assert buf.shape == (T,), name

    torch.testing.assert_close(diff.sqrt_alphas_bar.double(), alphas_bar.sqrt(), rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(diff.sqrt_one_minus_alphas_bar.double(), (1 - alphas_bar).sqrt(), rtol=1e-6, atol=1e-7)

    # the forward coefficients should preserve unit variance: a_bar + (1 - a_bar) = 1
    total = diff.sqrt_alphas_bar**2 + diff.sqrt_one_minus_alphas_bar**2
    torch.testing.assert_close(total, torch.ones(T), rtol=0, atol=1e-6)

    assert torch.all(diff.sqrt_alphas_bar[1:] < diff.sqrt_alphas_bar[:-1])
    assert diff.sqrt_alphas_bar[-1] ** 2 < 1e-4


def test_loss_reaches_every_parameter(tiny_unet):
    """It should return a finite scalar loss whose backward pass gives every UNet parameter a non-zero gradient"""
    model = wake_up(tiny_unet)
    diff = Gaussian_Diffusion(model, timesteps=1000)

    loss = diff.loss(torch.rand(2, 3, 16, 16) * 2.0 - 1.0)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()

    # None or all-zero grad => module was built but never used in forward
    for name, p in model.named_parameters():
        assert p.grad is not None, f"{name}: no grad"
        assert p.grad.abs().sum() > 0, f"{name}: zero grad"