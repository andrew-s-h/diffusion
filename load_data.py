import os
import json
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T 
import matplotlib
import time

matplotlib.use("Agg")
import matplotlib.pyplot as plt 

from datasets import load_dataset

REPO_ID = "flwrlabs/celeba"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "data", "celeba64")
IMG_SIZE = 64
CROP = 178
SEED = 0

BATCH = 512

NUM_WORKERS = min(8, max(1, (os.cpu_count() or 2) - 1))
FLUSH_FREQ = 16
SPLITS = ("train", "valid", "test")
NON_ATTR_COLUMNS = ("image", "celeb_id")

META_NAME = "meta.json" 

#resizing
composer = T.Compose([
    # crop from the center
    T.CenterCrop(CROP),
    T.Resize(IMG_SIZE),
    T.PILToTensor(),
])

# remember float = 4 byte = 32, double = 8 byte = 64 bit 
def _to_model_range(x_uint8: torch.Tensor) -> torch.Tensor:
    return x_uint8.float() / 127.5 - 1.0 

class _SplitDecode(torch.utils.data.Dataset):

    """
    Prepares one HF split for Dataloader. Used during build

    Args: 
        dataset: HF Dataset for one split; dataset[i]["image] -> PIL.Image
        transform (T.compose): PIL -> uint8(C, H, W) tensor
    """
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return self.dataset.num_rows

    def __getitem__(self, i: int) -> torch.Tensor:
        img = self.dataset[i]["image"].convert("RGB")
        return self.transform(img) # uint8(C, H, W)

class CelebaMemmap:
    """Preprocesses the data and persists each split as an on-disk uint8 memmap"""

    def __init__(self, out_dir: str = OUT_DIR, transform: T.Compose = composer, 
                 img_size: int = IMG_SIZE):
        self.out_dir = out_dir
        self.transform = transform
        self.img_size = img_size
        os.makedirs(out_dir, exist_ok=True)

    def _path(self, split: str) -> str:
        """returns path of out the directory containing split.npy"""
        return os.path.join(self.out_dir, f"{split}.npy")

    def build(self, dataset, splits=SPLITS, batch: int = BATCH,
                overwrite: bool = False) -> dict:
        """Processes every split to disk and writes a metadata manigest
            
        Transforms each split into a fixed-size uint8 memmap via _build_split and dumps
        one metadata file describing the whole dataset build. 

        Args: 
            dataset: HF DatasetDict. dataset[split] -> Dataset.
            splits (Iterable[str]): names of the splits to build. 
            batch (int): rows per flush for _build_split.
                Defaule BATCH
            overwrite(bool): existing split files are skipped if False.
                if True, split files are rebuilt
        
        Returns: 
            dict: Metadata written to META_NAME. 
                Keys:
                    "repo", "img_size", "crop", "seed", "layout", "dtype",
                    "value_range", and "splits" -> {split: {"path", "shape"}}.
        """
        meta = {
            "repo": REPO_ID,
            "img_size": self.img_size,
            "crop": CROP,
            "seed": SEED,
            "layout": "NCHW",
            "dtype": "uint8",
            "value_range": "[0, 255]", # rescale at load
            "splits":{}
            }
        for split in splits:
            meta["splits"][split] = self._build_split(dataset[split], split, batch, overwrite)
        with open(os.path.join(self.out_dir, META_NAME), "w") as f:
            json.dump(meta, f, indent=2)
        return meta

    # per individual split
    def _build_split(self, dataset, split: str, batch: int, overwrite: bool) -> dict:
        """
        Transforms split into fixed-size uint8 .npy memmap on disk

        streams split in chunks of 'batch', decodes to RGB, and applies self.transform, 
        and writes the result into slot i of the pre-allocated (n, 3, img)size, img_size)
        memmap. 

        Args:
            dataset: HF Dataset for one split; dataset[i]["image"] -> PIL.IMAGE,
                dataset.num_rows -> int
            split (str): split name that will be used for outout fname and logs
            batch (int): number of rows per chunk/flush
            overwrite (bool): if True, will overwrite existing files
        
            Returns:
                dict: {"path": <basename>.npy, "shape": [n, 3, img_size, img_size]}
            Writes:
                <out_dir>/<split>.npy: uint8array
        """
        path = self._path(split)
        n = dataset.num_rows
        shape = (n, 3, self.img_size, self.img_size)
    
        if os.path.exists(path) and not overwrite:
            print(f"[{split}] already exists")
            return {"path": os.path.basename(path), "shape": list(shape)}

        # provides iterable over dataset
        loader = DataLoader(
            _SplitDecode(dataset, self.transform),
            batch_size=batch,
            shuffle=False,
            num_workers=NUM_WORKERS,
        )   
        # e.g. we write to train.npy.tmp 
        tmp_path = path + ".tmp"

        mm = np.lib.format.open_memmap(
            tmp_path, mode="w+", dtype=np.uint8, shape=shape
            )
        
        t0 = time.time()
        row = 0
        try:
            for k, chunk in enumerate(loader):
                b = chunk.shape[0]
                mm[row:row + b] = chunk.numpy()
                row = row + b
                if (k+1) % FLUSH_FREQ == 0:
                    mm.flush()
                print(f"[{split}] {row}/{n} ({time.time() - t0:.1f}s)", flush=True)
            mm.flush()
        except BaseException:
            del mm
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise        
        del mm
        assert row == n, f"{split}: wrote {row} rows, expected {n}"
    
        os.replace(tmp_path, path)
        print(f"[{split}] wrote {path}  shape = {shape}")
        return {"path": os.path.basename(path), "shape": list(shape)}

    def open(self, split: str) -> np.memmap:
        """Reopens a saved split in red-only mode"""
        return np.load(self._path(split),mmap_mode='r')

    def dataset(self, split:str) -> "CelebaMemmapDataSet":
        return CelebaMemmapDataSet(self.open(split))

class CelebaMemmapDataSet(torch.utils.data.Dataset):
    """Serves rows from a memmaped split. Images served one at a time
    from a saved .npy split  
    
    Reads row i off disk on demand and converts uint8[0-255] -> float32 [-1, 1]

    Args:
        mm (np.memmap): a split oppened with CelebaMemmap.open(),
        shape (N, 3, H, W)
    
    Returns:
        torch.Tensor: float32 (3, H, W) w/values [-1, 1]
    """
    def __init__(self, mm: np.memmap):
        self.mm = mm

    def __len__(self) -> int:
        return self.mm.shape[0]

    def __getitem__(self, i: int) -> torch.Tensor:
        x = torch.from_numpy(np.array(self.mm[i]))
        return _to_model_range(x)

    
def preview_ds(ds, seed: int = SEED) -> None:
    """print the datasets structure and save one random face as png

    Args:()
        ds:DatasetDict from load_dataset(REPO_ID) with keys train/valid/test

    """
    print("splits:", list(ds.keys()))
    for split in SPLITS:
        print(f"    {split:<6} rows: {ds[split].num_rows} cols: {ds[split].num_columns}")

    print("\ncolumns:", ds["train"].num_columns, "total w/celeb_id & image")
    ds_attributes = [col_name for col_name in ds["train"].column_names if col_name not in NON_ATTR_COLUMNS]
    print(f"\n{len(ds_attributes)} attributes:")
    print(", ".join(ds_attributes))

    #check features same across splits
    for split in SPLITS:
        assert ds[split].features == ds["train"].features, f"{split} some features different than train"
    print("\nIdentical schema across splits! GOOD")
    
    generator = random.Random(seed)
    fig, axes = plt.subplots(1, len(SPLITS), figsize=(10, 3.6))

    for ax, split in zip(axes, SPLITS):
        i = generator.randrange(ds[split].num_rows)
        img = ds[split][i]["image"]
        x = composer(img) # (C, H, W)

        ax.imshow(x.permute(1, 2, 0).numpy()) #(H, W, C)
        ax.set_title(f"{split} [{i}]")
        ax.axis("off")

    # sanity check: should print -1.0 and 1.0
    xm = _to_model_range(x)
    print("\nmodel range:", xm.min().item(), xm.max().item())

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "sample_per_split.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("wrote", path)


if  __name__ == "__main__":
    ds  = load_dataset(REPO_ID)
    preview_ds(ds)

    prep = CelebaMemmap()
    prep.build(ds)

    train = prep.dataset("train")
    x0 = train[0]
    
    print("train:", len(train), "example:", tuple(x0.shape),
          x0.dtype, "range", (x0.min().item(), x0.max().item()))