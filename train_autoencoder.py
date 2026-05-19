import os
os.environ["HF_HOME"] = "./hf_cache"

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from accelerate import Accelerator
from tqdm import tqdm
import argparse
import numpy as np
from autoencoder import JSCC_Autoencoder

# ----------------------------------------------------------------------
# Cached Dataset — loads precomputed lambda_val from disk (no forward model at runtime)
# Run precompute_lambda.py once to populate ./lambda_cache/ before training.
# ----------------------------------------------------------------------
class HHDCCachedDataset(Dataset):
    def __init__(self, cache_path, shape_path):
        shape = tuple(np.load(shape_path).tolist())
        self.data = np.memmap(cache_path, dtype="float32", mode="r", shape=shape)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        lv = torch.from_numpy(self.data[idx].copy())
        return lv, lv   # (lambda_val, lambda_val) — noise applied on GPU in training loop


def log_mse_loss(pred, target, eps=1e-4):
    return F.mse_loss(torch.log(pred + eps), torch.log(target + eps))


def unpack_batch(batch):
    if not isinstance(batch, (list, tuple)) or len(batch) < 2:
        raise ValueError("Expected each batch to contain at least input_data and output_data.")
    return batch[0], batch[1]


# =================================================================================
# Configuration
# =================================================================================
def get_config(compression_ratio: int):
    return {
        "input_shape": (128, 32, 16),
        "base_channels": 32,
        "num_res_blocks": 6,
        "num_res_blocks_lidar": 1,
        "channel_snr_db_range": (-5, 20),
        "compression_ratio": compression_ratio,
        "learning_rate": 1e-4,
        "batch_size": 128,
        "num_epochs": 500,
        "results_dir": "./autoencoder_lambda",
        "noise_eta": 0.5,       # Gaussian readout noise std (must match precompute_lambda.py)
        "lambda_cache": "./lambda_cache",
    }


# =================================================================================
# Training Loop
# =================================================================================
def train(config):
    accelerator = Accelerator(mixed_precision="bf16")

    os.makedirs(config["results_dir"], exist_ok=True)

    model = JSCC_Autoencoder(config)

    warmup_epochs = 10
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=config["num_epochs"] - warmup_epochs, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])

    # Datasets (precomputed lambda_val — run precompute_lambda.py first)
    cache = config["lambda_cache"]
    train_dataset = HHDCCachedDataset(
        os.path.join(cache, "train.npy"),
        os.path.join(cache, "train_shape.npy"),
    )
    val_dataset = HHDCCachedDataset(
        os.path.join(cache, "validation.npy"),
        os.path.join(cache, "validation_shape.npy"),
    )

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=config["batch_size"]*2, shuffle=False, num_workers=0)

    criterion = log_mse_loss

    model, optimizer, scheduler, train_loader, val_loader = accelerator.prepare(
        model, optimizer, scheduler, train_loader, val_loader
    )

    # Resume from best validation checkpoint if exists
    best_train_loss = float('inf')
    best_val_loss = float('inf')
    resume_checkpoint_path = os.path.join(
        config["results_dir"],
        f"best_val_model_compression_{config['compression_ratio']}.pt"
    )

    start_epoch = 0
    if os.path.exists(resume_checkpoint_path):
        if accelerator.is_local_main_process:
            print(f"Resuming from {resume_checkpoint_path}")
        checkpoint = torch.load(resume_checkpoint_path, map_location='cpu')
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.load_state_dict(checkpoint['model_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        best_train_loss = checkpoint.get('best_train_loss', float('inf'))

    # ====================== TRAINING ======================
    for epoch in range(start_epoch, config["num_epochs"]):
        model.train()
        total_train_loss = 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} Train", disable=not accelerator.is_local_main_process)

        for batch in train_pbar:
            lambda_val, _ = unpack_batch(batch)
            noisy = torch.poisson(lambda_val) + torch.randn_like(lambda_val) * config["noise_eta"]
            input_data = noisy.unsqueeze(1)                        # (B, 1, D, H, W)
            output_data = lambda_val.unsqueeze(1)                  # (B, 1, D, H, W) — match model output

            optimizer.zero_grad()

            snr_db = torch.rand(input_data.size(0), 1, device=accelerator.device) * \
                     (config["channel_snr_db_range"][1] - config["channel_snr_db_range"][0]) + \
                     config["channel_snr_db_range"][0]

            with accelerator.autocast():
                reconstructed_data = model(input_data, snr_db)
                loss = criterion(reconstructed_data, output_data)

            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_train_loss += loss.item()
            train_pbar.set_postfix(loss=f"{loss.item():.6f}")

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        model.eval()
        total_val_loss = 0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} Val", disable=not accelerator.is_local_main_process)
        with torch.no_grad():
            for batch in val_pbar:
                lambda_val, _ = unpack_batch(batch)
                noisy = torch.poisson(lambda_val) + torch.randn_like(lambda_val) * config["noise_eta"]
                input_data = noisy.unsqueeze(1)
                output_data = lambda_val.unsqueeze(1)              # (B, 1, D, H, W) — match model output
                snr_db = torch.full(
                    (input_data.size(0), 1),
                    (config["channel_snr_db_range"][0] + config["channel_snr_db_range"][1]) / 2.0,
                    device=accelerator.device,
                )
                with accelerator.autocast():
                    reconstructed_data = model(input_data, snr_db)
                    loss = criterion(reconstructed_data, output_data)
                total_val_loss += loss.item()
                val_pbar.set_postfix(loss=f"{loss.item():.6f}")

        avg_val_loss = total_val_loss / len(val_loader)
        scheduler.step()

        if accelerator.is_local_main_process:
            print(f"Epoch {epoch+1}: Train {avg_train_loss:.6f} | Val {avg_val_loss:.6f} | LR {scheduler.get_last_lr()[0]:.2e}")

        # Save best validation model
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(config["results_dir"], f"best_val_model_compression_{config['compression_ratio']}.pt")
            accelerator.save({
                'epoch': epoch,
                'model_state_dict': unwrapped_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'best_train_loss': best_train_loss,
            }, save_path)
            if accelerator.is_local_main_process:
                print(f"New best validation model saved (Loss: {best_val_loss:.6f})")

    print("Training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D Autoencoder Training")
    parser.add_argument("--compression_ratio", type=int, default=16)
    args = parser.parse_args()
    print(f"Compression ratio: {args.compression_ratio}")

    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)

    torch.set_num_threads(1)
    CONFIG = get_config(args.compression_ratio)
    train(CONFIG)
