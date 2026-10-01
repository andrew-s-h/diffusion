"""Trains the DDPM eps_theta U-Net on 64x64 CelebA with L_simple.

Samples and saved weights come from the EMA copy, following Ho et al.
"""
import json
import math
import os
import time

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from ddpm import EMA, Gaussian_Diffusion, UNet
from load_data import CelebaMemmap

STEPS = 50000
BATCH_SIZE = 24
LR = 2e-4
WARMUP = 2000
GRAD_CLIP = 1.0
EMA_DECAY = 0.9995

LOG_EVERY = 250
SAMPLE_EVERY = 2500
CKPT_EVERY = 1000
N_SAMPLES = 6
OUT_DIR = os.path.join("runs", "ddpm_50k")
RESUME = None


def pick_device() -> str:
    """cuda -> mps -> cpu"""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train() -> None:
    torch.manual_seed(0)
    device = pick_device()
    os.makedirs(OUT_DIR, exist_ok=True)

    # num_workers=0: on macOS each worker would get its own ~2 GB copy of the memmap
    data = CelebaMemmap().dataset("train")
    loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)

    diffusion = Gaussian_Diffusion(UNet())
    diffusion.to(device)
    model = diffusion.model

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    # linear warmup to LR, then constant
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / WARMUP))

    ema = EMA(model, decay=EMA_DECAY)
    # second wrapper around the EMA net; ema.update edits it in place, so it
    # always samples with the current EMA weights
    ema_diffusion = Gaussian_Diffusion(ema.shadow)
    ema_diffusion.to(device)

    step = 0
    if RESUME is not None:
        ck = torch.load(RESUME, map_location=device)
        model.load_state_dict(ck["model"])
        ema.shadow.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        step = ck["step"]
        print(f"resumed at step {step}")

    def save_ckpt():
        state = {
            "model": model.state_dict(),
            "ema": ema.shadow.state_dict(),
            "opt": opt.state_dict(),
            "sched": sched.state_dict(),
            "step": step,
        }
        # atomic: a crash mid-write leaves the previous last.pt intact
        path = os.path.join(OUT_DIR, "last.pt")
        torch.save(state, path + ".tmp")
        os.replace(path + ".tmp", path)

    # one JSON object per line, appended, so a crash or resume never loses earlier entries
    log_path = os.path.join(OUT_DIR, "log.jsonl")

    model.train()
    running = 0.0
    t0 = time.time()
    while step < STEPS:
        for x_0 in loader:
            x_0 = x_0.to(device)

            loss = diffusion.loss(x_0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            sched.step()

            # EMA warmup: low decay early so the average forgets the random init
            ema.decay = min(EMA_DECAY, (1 + step) / (10 + step))
            ema.update(model)
            step += 1

            running += loss.item()
            if step % LOG_EVERY == 0:
                print(f"step {step:>7d} | loss {running / LOG_EVERY:.4f} | {time.time() - t0:.0f}s", flush=True)
                with open(log_path, "a") as f:
                    f.write(json.dumps({"step": step, "loss": running / LOG_EVERY,
                                        "lr": sched.get_last_lr()[0], "ema_decay": ema.decay,
                                        "elapsed_s": round(time.time() - t0, 1)}) + "\n")
                running = 0.0

            if step % SAMPLE_EVERY == 0:
                x = ema_diffusion.sample((N_SAMPLES, 3, 64, 64))
                x = (x.clamp(-1.0, 1.0) + 1.0) / 2.0
                save_image(x, os.path.join(OUT_DIR, f"samples_{step:07d}.png"), nrow=int(math.sqrt(N_SAMPLES)))

            if step % CKPT_EVERY == 0:
                save_ckpt()

            if step >= STEPS:
                break

    save_ckpt()


if __name__ == "__main__":
    train()