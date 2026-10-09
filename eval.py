"""Evaluates DDPM checkpoints on 64x64 CelebA.

Three evaluations, each toggled by its RUN_* flag:

    denoising loss   L_simple on fixed (x_0, t, eps) triples from the valid and
                     train splits, overall and binned by t.
    sample grids     one grid per checkpoint from a fixed seed.
    FID              generated samples against the full CelebA train split.

Checkpoint tags:

    ema_0400000      EMA weights from ema_0400000.pt
    raw_0400000      live weights from chkpt_400000.pt["model"]

Results accumulate in EVAL_DIR/results.json keyed by tag, and finished entries
are skipped on rerun. FID samples are cached as uint8 chunks, so an interrupted
run resumes where it stopped.
"""
import json
import math
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
from torchvision.utils import save_image

from ddpm import Gaussian_Diffusion, UNet
from load_data import CelebaMemmap
from train import pick_device

RUN_DIR = os.path.join("runs", "ddpm_150k")
EVAL_DIR = os.path.join(RUN_DIR, "eval")
SEED = 0

EMA_TAGS = [f"ema_{s:07d}" for s in range(75_000, 400_001, 25_000)]
RAW_TAGS = [f"raw_{s:07d}" for s in (100_000, 163_000, 250_000, 400_000)]

RUN_LOSS = True
LOSS_TAGS = EMA_TAGS + RAW_TAGS
LOSS_N = 5_000
LOSS_BATCH = 250
T_BINS = 10

RUN_GRIDS = True
GRID_TAGS = ["ema_0400000"]
GRID_N = 25

RUN_FID = True
FID_TAGS = ["ema_0400000", "ema_0200000"]
FID_N = 10_000
SAMPLE_BATCH = 250
INCEPTION_BATCH = 250


def load_diffusion(tag: str, device: str) -> Gaussian_Diffusion:
    """Builds a Gaussian_Diffusion around the U-Net weights named by tag.

    The schedule isn't stored in the checkpoints; it comes from
    Gaussian_Diffusion's defaults, which train.py also used. The network is put
    in eval mode so dropout is off.

    Args:
        tag (str): "ema_<7-digit step>" or "raw_<7-digit step>".
        device (str): device to place the model on.

    Returns:
        Gaussian_Diffusion: loaded model in eval mode on device.
    """
    kind, step = tag.split("_")
    if kind == "ema":
        path = os.path.join(RUN_DIR, f"ema_{step}.pt")
        state = torch.load(path, map_location="cpu")
    elif kind == "raw":
        path = os.path.join(RUN_DIR, f"chkpt_{int(step)}.pt")
        ck = torch.load(path, map_location="cpu")
        state = ck["model"]
    else:
        raise ValueError(f"unknown tag kind {kind!r}; expected 'ema' or 'raw'")

    net = UNet()
    net.load_state_dict(state)
    net.eval()
    diffusion = Gaussian_Diffusion(net)
    diffusion.to(device)
    return diffusion


def to_uint8(x: torch.Tensor) -> np.ndarray:
    """Converts [-1, 1] images to uint8, inverting load_data._to_model_range.

    Generated and real images then reach Inception in the same format. Values
    are clamped first because the final reverse step can land slightly outside
    [-1, 1].

    Args:
        x (torch.Tensor): (B, 3, H, W) images, approximately in [-1, 1].

    Returns:
        np.ndarray: (B, 3, H, W) uint8 array on the CPU.
    """
    x = x.clamp(-1.0, 1.0)
    x = (x + 1.0) * 127.5
    x = x.round()
    x = x.to(torch.uint8)
    x = x.cpu()
    return x.numpy()


@torch.no_grad()
def denoising_loss(diffusion: Gaussian_Diffusion, split: str, n: int, device: str) -> dict:
    """Computes L_simple on n evenly spaced images from one split.

    Batch i draws t and eps from a CPU generator seeded with SEED + i, so every
    checkpoint is scored on identical (x_0, t, eps) triples. Diffusion.loss
    isn't used because it draws from the global RNG.

    Args:
        diffusion (Gaussian_Diffusion): model in eval mode.
        split (str): "train", "valid" or "test".
        n (int): number of images; at most the split size.
        device (str): device the model is on.

    Returns:
        dict: "loss" -> mean per-image MSE over all n images, and
            "loss_by_t" -> T_BINS means, bin k covering
            t in [k*T/T_BINS, (k+1)*T/T_BINS).
    """
    data = CelebaMemmap().dataset(split)
    idx = np.linspace(0, len(data), n, endpoint=False)
    idx = idx.astype(int)
    loader = DataLoader(Subset(data, idx.tolist()), batch_size=LOSS_BATCH, shuffle=False)

    T = diffusion.timesteps
    sums = np.zeros(T_BINS)
    counts = np.zeros(T_BINS)

    for i, x_0 in enumerate(loader):
        g = torch.Generator()
        g.manual_seed(SEED + i)
        b = x_0.shape[0]
        t = torch.randint(0, T, (b,), generator=g)
        eps = torch.randn(x_0.shape, generator=g)

        x_0 = x_0.to(device)
        t = t.to(device)
        eps = eps.to(device)

        x_t = diffusion.q_sample(x_0, t, eps)
        eps_hat = diffusion.model(x_t, t)
        per_image = ((eps_hat - eps) ** 2).mean(dim=(1, 2, 3))

        bins = (t * T_BINS) // T
        np.add.at(sums, bins.cpu().numpy(), per_image.cpu().numpy())
        np.add.at(counts, bins.cpu().numpy(), 1)

    return {
        "loss": float(sums.sum() / counts.sum()),
        "loss_by_t": (sums / counts).tolist(),
    }


@torch.no_grad()
def save_grid(diffusion: Gaussian_Diffusion, path: str) -> None:
    """Samples GRID_N images from seed SEED and saves them as one grid.

    sample() makes the same RNG calls regardless of the weights, so grid
    position k gets the same x_T and per-step noise for every checkpoint.

    Args:
        diffusion (Gaussian_Diffusion): model in eval mode.
        path (str): output .png path.
    """
    torch.manual_seed(SEED)
    x = diffusion.sample((GRID_N, 3, 64, 64))
    x = x.clamp(-1.0, 1.0)
    x = (x + 1.0) / 2.0
    save_image(x, path, nrow=int(math.sqrt(GRID_N)))


@torch.no_grad()
def generate_samples(diffusion: Gaussian_Diffusion, tag: str, n: int) -> np.ndarray:
    """Returns n samples for FID, generating only the chunks not already on disk.

    Chunk i is SAMPLE_BATCH images drawn with seed SEED + i, saved under
    EVAL_DIR/samples/<tag>_b<SAMPLE_BATCH>/. Every tag uses the same seeds, so
    FID differences between tags don't come from different noise. Each chunk is
    written to a temp file and renamed, so a crash can't leave a partial chunk.

    Args:
        diffusion (Gaussian_Diffusion): model in eval mode.
        tag (str): checkpoint tag; names the cache directory.
        n (int): number of samples.

    Returns:
        np.ndarray: (n, 3, 64, 64) uint8 samples.
    """
    out_dir = os.path.join(EVAL_DIR, "samples", f"{tag}_b{SAMPLE_BATCH}")
    os.makedirs(out_dir, exist_ok=True)
    n_chunks = math.ceil(n / SAMPLE_BATCH)

    chunks = []
    for i in range(n_chunks):
        path = os.path.join(out_dir, f"chunk_{i:04d}.npy")
        if os.path.exists(path):
            chunks.append(np.load(path))
            continue

        start = time.time()
        torch.manual_seed(SEED + i)
        x = diffusion.sample((SAMPLE_BATCH, 3, 64, 64))
        chunk = to_uint8(x)

        tmp_path = os.path.join(out_dir, f"chunk_{i:04d}.tmp.npy")
        np.save(tmp_path, chunk)
        os.replace(tmp_path, path)
        chunks.append(chunk)

        per_chunk = time.time() - start
        hours_left = (n_chunks - i - 1) * per_chunk / 3600
        print(f"[{tag}] chunk {i + 1}/{n_chunks} | {per_chunk:.0f}s/chunk | ~{hours_left:.1f}h left", flush=True)

    images = np.concatenate(chunks)
    return images[:n]


@torch.no_grad()
def inception_features(images: np.ndarray, inception: FeatureExtractorInceptionV3, device: str) -> np.ndarray:
    """Embeds uint8 images as 2048-d Inception-v3 pool features.

    Uses torch-fidelity's "inception-v3-compat" extractor: the TensorFlow FID
    weights with a TF1-matching bilinear resize to 299x299, so features match
    the reference FID implementation. It takes uint8 input directly.

    Args:
        images (np.ndarray): (N, 3, H, W) uint8; a read-only memmap is fine
            since each batch is copied.
        inception (FeatureExtractorInceptionV3): extractor in eval mode on device.
        device (str): device the extractor is on.

    Returns:
        np.ndarray: (N, 2048) float64 features.
    """
    feats = []
    for start in range(0, len(images), INCEPTION_BATCH):
        batch = np.array(images[start:start + INCEPTION_BATCH])
        x = torch.from_numpy(batch)
        x = x.to(device)
        out = inception(x)[0]
        feats.append(out.cpu().numpy())
    feats = np.concatenate(feats)
    return feats.astype(np.float64)


def feature_stats(feats: np.ndarray) -> tuple:
    """Fits a Gaussian to features: sample mean and unbiased covariance.

    Args:
        feats (np.ndarray): (N, D) features.

    Returns:
        tuple[np.ndarray, np.ndarray]: mean (D,) and covariance (D, D).
    """
    mu = feats.mean(axis=0)
    sigma = np.cov(feats, rowvar=False)
    return mu, sigma


def reference_stats(inception: FeatureExtractorInceptionV3, device: str) -> tuple:
    """Returns Inception statistics of the full train split.

    Reads the same uint8 memmap training used, so crop and resize match the
    training data. Computed once and cached at EVAL_DIR/ref_stats_train.npz.

    Args:
        inception (FeatureExtractorInceptionV3): extractor in eval mode on device.
        device (str): device the extractor is on.

    Returns:
        tuple[np.ndarray, np.ndarray]: mean (2048,) and covariance (2048, 2048).
    """
    path = os.path.join(EVAL_DIR, "ref_stats_train.npz")
    if os.path.exists(path):
        ref = np.load(path)
        return ref["mu"], ref["sigma"]

    images = CelebaMemmap().open("train")
    feats = inception_features(images, inception, device)
    mu, sigma = feature_stats(feats)
    np.savez(path, mu=mu, sigma=sigma, n=len(feats))
    return mu, sigma


def frechet_distance(mu_r: np.ndarray, sigma_r: np.ndarray,
                     mu_g: np.ndarray, sigma_g: np.ndarray) -> float:
    """Computes FID, the squared Frechet distance between two Gaussians.

        d^2 = ||mu_r - mu_g||^2 + Tr(S_r) + Tr(S_g) - 2 Tr((S_r S_g)^(1/2))

    S_r S_g has the same eigenvalues as the symmetric PSD matrix
    M = S_r^(1/2) S_g S_r^(1/2), so Tr((S_r S_g)^(1/2)) = sum_i sqrt(lambda_i(M)).
    This uses two symmetric eigendecompositions instead of scipy.linalg.sqrtm,
    which can return spurious imaginary parts. Round-off negatives are clipped
    to 0.

    Args:
        mu_r (np.ndarray): (D,) reference mean.
        sigma_r (np.ndarray): (D, D) reference covariance.
        mu_g (np.ndarray): (D,) generated mean.
        sigma_g (np.ndarray): (D, D) generated covariance.

    Returns:
        float: FID.
    """
    diff = mu_r - mu_g

    w_r, v_r = np.linalg.eigh(sigma_r)
    w_r = np.clip(w_r, 0.0, None)
    sqrt_sigma_r = (v_r * np.sqrt(w_r)) @ v_r.T

    m = sqrt_sigma_r @ sigma_g @ sqrt_sigma_r
    m = (m + m.T) / 2.0
    w_m = np.linalg.eigvalsh(m)
    w_m = np.clip(w_m, 0.0, None)
    tr_covmean = np.sqrt(w_m).sum()

    return float(diff @ diff + np.trace(sigma_r) + np.trace(sigma_g) - 2.0 * tr_covmean)


def fid(diffusion: Gaussian_Diffusion, tag: str, n: int,
        inception: FeatureExtractorInceptionV3, ref: tuple, device: str) -> float:
    """Computes FID for one checkpoint from n generated samples.

    Scores are only comparable at equal n.

    Args:
        diffusion (Gaussian_Diffusion): model in eval mode.
        tag (str): checkpoint tag; names the sample cache.
        n (int): number of generated samples.
        inception (FeatureExtractorInceptionV3): extractor in eval mode on device.
        ref (tuple): (mu, sigma) from reference_stats.
        device (str): device the extractor is on.

    Returns:
        float: FID.
    """
    images = generate_samples(diffusion, tag, n)
    feats = inception_features(images, inception, device)
    mu_g, sigma_g = feature_stats(feats)
    return frechet_distance(ref[0], ref[1], mu_g, sigma_g)


def load_results() -> dict:
    """Reads EVAL_DIR/results.json, or returns an empty dict if it doesn't exist."""
    path = os.path.join(EVAL_DIR, "results.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def save_results(results: dict) -> None:
    """Writes results.json via a temp file and rename, so a crash keeps the previous file."""
    path = os.path.join(EVAL_DIR, "results.json")
    with open(path + ".tmp", "w") as f:
        json.dump(results, f, indent=2)
    os.replace(path + ".tmp", path)


def main() -> None:
    device = pick_device()
    grid_dir = os.path.join(EVAL_DIR, "grids")
    os.makedirs(grid_dir, exist_ok=True)
    results = load_results()

    if RUN_LOSS:
        for tag in LOSS_TAGS:
            entry = results.setdefault(tag, {})
            diffusion = None
            for split in ("valid", "train"):
                key = f"loss_{split}_{LOSS_N}"
                if key in entry:
                    continue
                if diffusion is None:
                    diffusion = load_diffusion(tag, device)
                entry[key] = denoising_loss(diffusion, split, LOSS_N, device)
                save_results(results)
            valid = entry[f"loss_valid_{LOSS_N}"]["loss"]
            train = entry[f"loss_train_{LOSS_N}"]["loss"]
            print(f"[{tag}] loss valid {valid:.5f} | train {train:.5f} | gap {valid - train:+.5f}", flush=True)

    if RUN_GRIDS:
        for tag in GRID_TAGS:
            path = os.path.join(grid_dir, f"{tag}.png")
            if os.path.exists(path):
                continue
            diffusion = load_diffusion(tag, device)
            save_grid(diffusion, path)
            print(f"[{tag}] wrote {path}", flush=True)

    if RUN_FID:
        inception = FeatureExtractorInceptionV3("inception-v3-compat", ["2048"])
        inception.to(device)
        inception.eval()
        ref = reference_stats(inception, device)
        for tag in FID_TAGS:
            entry = results.setdefault(tag, {})
            key = f"fid_{FID_N}"
            if key not in entry:
                diffusion = load_diffusion(tag, device)
                entry[key] = fid(diffusion, tag, FID_N, inception, ref, device)
                save_results(results)
            print(f"[{tag}] FID@{FID_N} {entry[key]:.2f}", flush=True)


if __name__ == "__main__":
    main()