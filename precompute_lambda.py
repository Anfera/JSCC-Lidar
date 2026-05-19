import os
os.environ["HF_HOME"] = "./hf_cache"

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import load_dataset
from forward_model import LidarForwardImagingModel

CACHE_DIR = "./lambda_cache"
BATCH_SIZE = 64
NUM_WORKERS = 8

FORWARD_MODEL_KWARGS = dict(
    output_res_m=(3.0, 6.0),
    footprint_diameter_m=10.0,
    b=0.1,
    eta=0.5,
    ref_altitude=500.0,
    ref_photon_count=20.0,
)


def precompute_split(split: str, hf_ds, forward_model, device):
    hf_ds.set_format(type="torch", columns=["cube"])
    loader = DataLoader(hf_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)

    # Determine output shape from one batch
    sample_cube = hf_ds[0]["cube"].unsqueeze(0).to(device)
    with torch.no_grad():
        sample_lv = forward_model.compute_lambda(sample_cube)
    lv_shape = tuple(sample_lv.squeeze(0).shape)   # (D, H, W)

    N = len(hf_ds)
    out_path = os.path.join(CACHE_DIR, f"{split}.npy")
    shape_path = os.path.join(CACHE_DIR, f"{split}_shape.npy")

    fp = np.memmap(out_path, dtype="float32", mode="w+", shape=(N, *lv_shape))

    idx = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Precomputing {split}"):
            cubes = batch["cube"].to(device)
            lv = forward_model.compute_lambda(cubes)   # (B, D, H, W)
            b = lv.shape[0]
            fp[idx : idx + b] = lv.cpu().numpy()
            idx += b

    fp.flush()
    np.save(shape_path, np.array([N, *lv_shape]))
    print(f"Saved {split}: {out_path}  shape={N, *lv_shape}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(CACHE_DIR, exist_ok=True)

    forward_model = LidarForwardImagingModel(**FORWARD_MODEL_KWARGS).to(device)
    forward_model.eval()

    for split in tqdm(["train", "validation"], desc="Splits"):
        hf_ds = load_dataset("anfera236/HHDC", split=split)
        precompute_split(split, hf_ds, forward_model, device)


if __name__ == "__main__":
    main()
