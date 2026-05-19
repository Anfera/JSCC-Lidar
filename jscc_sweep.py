import sys
import os
import gc
import json
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from autoencoder import JSCC_Autoencoder
from forward_model import LidarForwardImagingModel
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from torch.distributions import Poisson
from tqdm import tqdm
import lpips
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from canopyPlots import createCHM

# ── Config ────────────────────────────────────────────────────────────────────
COMPRESSION_RATIO = 8
CSNR_RANGE = range(-5, 21)
NUM_RUNS = 50

CONFIG = {
    "input_shape": (128, 32, 16),
    "base_channels": 32,
    "num_res_blocks": 6,
    "num_res_blocks_lidar": 1,
    "channel_snr_db_range": (-5, 20),
    "compression_ratio": COMPRESSION_RATIO,
    "learning_rate": 1e-4,
    "batch_size": 64,
    "num_epochs": 200,
    "results_dir": "./autoencoder_lambda",
}

DEVICE = 'cuda'
RESOLUTION = 2
FACTOR = 3
SAMPLING_TIMESTEPS = 250
IMAGE_SIZE = 96 // RESOLUTION   # 48
LR_MULTIPLIER = 50000
EPSILON = 1e-6
RECON_THRESHOLD = 0.05
ETA_DIFFUSION = 1.0
RECOMPUTE_METRICS = False

base_result_path = './resultCubes/'
os.makedirs(base_result_path, exist_ok=True)

# ── Load JSCC autoencoder (once) ──────────────────────────────────────────────
model = JSCC_Autoencoder(CONFIG)
resume_checkpoint_path = os.path.join(
    CONFIG["results_dir"],
    f"best_val_model_compression_{CONFIG['compression_ratio']}.pt"
)
if os.path.exists(resume_checkpoint_path):
    print(f"Loading checkpoint: {resume_checkpoint_path}")
    checkpoint = torch.load(resume_checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded epoch {checkpoint.get('epoch', 'N/A')} | best_val_loss={checkpoint.get('best_val_loss', 'N/A')}")
else:
    print(f"No checkpoint found at: {resume_checkpoint_path}")

# ── Forward model (once) ──────────────────────────────────────────────────────
forward_model = LidarForwardImagingModel(
    output_res_m=(3.0, 6.0),
    footprint_diameter_m=10.0,
    b=0.1,
    eta=0.5,
    ref_altitude=500.0,
    ref_photon_count=20.0,
)

# ── Load ground truth (once) ──────────────────────────────────────────────────
def load_and_preprocess_data():
    """Load, crop, and normalize the ground truth voxel cube."""
    ground_truth = torch.from_numpy(np.swapaxes(np.load('TestCube/southeast.npy'), -1, -2)).float().to(DEVICE)
    ground_truth = ground_truth[:, :(96 // RESOLUTION) * FACTOR, :(96 // RESOLUTION) * FACTOR]
    ground_truth = ground_truth / (ground_truth.max(dim=0)[0] + EPSILON)
    ground_truth = torch.roll(ground_truth, shifts=20, dims=0)
    ground_truth = torch.flip(ground_truth, dims=[1, 2])
    return ground_truth
    
clean = load_and_preprocess_data()

ground_truth = load_and_preprocess_data()

forward_model = forward_model.to(DEVICE)
clean_gpu = clean.to(DEVICE)  # fed to forward_model (GPU) for per-run Poisson draws

# ── Diffusion model (once) ────────────────────────────────────────────────────
model_unet = Unet(
    dim=128,
    dim_mults=(8, 16, 16, 16),
    flash_attn=True,
    channels=128
)
diffusion = GaussianDiffusion(
    model_unet,
    image_size=IMAGE_SIZE,
    timesteps=1000,
    sampling_timesteps=SAMPLING_TIMESTEPS
)
trainer = Trainer(
    diffusion,
    train_batch_size=8,
    train_lr=8e-5,
    train_num_steps=700000,
    gradient_accumulate_every=8,
    ema_decay=0.995,
    amp=True,
    resolution=RESOLUTION
)
trainer.load(0)
trainer.ema.ema_model.eval()


def unet_wrapper(x, time_cond):
    return trainer.ema.ema_model.model_predictions(
        x, time_cond, x_self_cond=None, clip_x_start=True, rederive_pred_noise=True
    )


# Precompute DDIM schedule (once)
times = torch.linspace(-1, 1000 - 1, steps=SAMPLING_TIMESTEPS + 1)
times = list(reversed(times.int().tolist()))
time_pairs = list(zip(times[:-1], times[1:]))

lpips_fn = lpips.LPIPS(net='vgg')


def save_results(input_image: torch.Tensor,
                 sample: np.ndarray,
                 gt_image: torch.Tensor,
                 save_path: str) -> dict:
    input_np = input_image.cpu().numpy()
    sample_np = sample
    gt_np = gt_image.cpu().numpy()

    _, dtm_input, hillshade_input, chm_input = createCHM(input_np, porcentaje=0.95)
    _, dtm_recon, hillshade_recon, chm_recon = createCHM(sample_np, porcentaje=0.95)
    _, dtm_gt, hillshade_gt, chm_gt = createCHM(gt_np, porcentaje=0.95)

    chm_input = chm_input * 0.5
    chm_recon = chm_recon * 0.5
    chm_gt = chm_gt * 0.5
    dtm_input = dtm_input * 0.5
    dtm_recon = dtm_recon * 0.5
    dtm_gt = dtm_gt * 0.5

    fig, axs = plt.subplots(3, 4, figsize=(15, 15))

    axs[0, 0].imshow(chm_input, cmap='viridis')
    axs[0, 0].set_aspect(0.5)
    axs[0, 0].set_title('CHM Input')

    axs[0, 1].imshow(chm_recon, cmap='viridis', vmin=chm_gt.min(), vmax=chm_gt.max())
    axs[0, 1].set_title('CHM Reconstruction')

    axs[0, 2].imshow(chm_gt, cmap='viridis')
    axs[0, 2].set_title('CHM Ground Truth')

    axs[1, 0].imshow(dtm_input, cmap='copper')
    axs[1, 0].imshow(hillshade_input, cmap='Grays', alpha=0.35)
    axs[1, 0].set_aspect(0.5)
    axs[1, 0].set_title('DTM Input')

    axs[1, 1].imshow(dtm_recon, cmap='copper', vmin=dtm_gt.min(), vmax=dtm_gt.max())
    axs[1, 1].imshow(hillshade_recon, cmap='Grays', alpha=0.35)
    axs[1, 1].set_title('DTM Reconstruction')

    axs[1, 2].imshow(dtm_gt, cmap='copper')
    axs[1, 2].imshow(hillshade_gt, cmap='Grays', alpha=0.35)
    axs[1, 2].set_title('DTM Ground Truth')

    profile_index_input = 3 * input_np.shape[1] // 4
    profile_index_gt = 3 * gt_np.shape[1] // 4

    axs[2, 0].imshow(input_np[::-1, profile_index_input, :], cmap='gray_r', interpolation='nearest')
    axs[2, 0].set_aspect(1 / 3)
    axs[2, 0].set_title('Profile Input')

    axs[2, 1].imshow(sample_np[::-1, profile_index_gt, :], cmap='gray_r', interpolation='nearest')
    axs[2, 1].set_title('Profile Reconstruction')

    axs[2, 2].imshow(gt_np[::-1, profile_index_gt, :], cmap='gray_r', interpolation='nearest')
    axs[2, 2].set_title('Profile Ground Truth')

    error_chm = np.abs(chm_recon - chm_gt)
    error_dtm = np.abs(dtm_recon - dtm_gt)
    error_profile = np.abs(sample_np[::-1, profile_index_gt, :] - gt_np[::-1, profile_index_gt, :])

    im0 = axs[0, 3].imshow(error_chm, cmap='turbo')
    axs[0, 3].set_title('CHM Error')
    fig.colorbar(im0, ax=axs[0, 3])

    im1 = axs[1, 3].imshow(error_dtm, cmap='turbo')
    axs[1, 3].set_title('DTM Error')
    fig.colorbar(im1, ax=axs[1, 3])

    im2 = axs[2, 3].imshow(error_profile, cmap='turbo')
    axs[2, 3].set_title('Profile Error')
    fig.colorbar(im2, ax=axs[2, 3])

    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()

    ssim_chm = ssim(chm_gt, chm_recon, data_range=chm_gt.max() - chm_gt.min(), win_size=13)
    psnr_chm = psnr(chm_gt, chm_recon, data_range=chm_gt.max() - chm_gt.min())
    ssim_dtm = ssim(dtm_gt, dtm_recon, data_range=dtm_gt.max() - dtm_gt.min(), win_size=13)
    psnr_dtm = psnr(dtm_gt, dtm_recon, data_range=dtm_gt.max() - dtm_gt.min())

    lpips_chm = lpips_fn.forward(
        torch.tensor(chm_gt).float().unsqueeze(0).unsqueeze(0),
        torch.tensor(chm_recon).float().unsqueeze(0).unsqueeze(0)
    ).item()
    lpips_dtm = lpips_fn.forward(
        torch.tensor(dtm_gt).float().unsqueeze(0).unsqueeze(0),
        torch.tensor(dtm_recon).float().unsqueeze(0).unsqueeze(0)
    ).item()

    mse = torch.mean((torch.tensor(gt_np) - torch.tensor(sample_np)) ** 2).item()
    mae = torch.mean(torch.abs(torch.tensor(gt_np) - torch.tensor(sample_np))).item()

    return {
        "ssim_chm": ssim_chm,
        "psnr_chm": psnr_chm,
        "ssim_dtm": ssim_dtm,
        "psnr_dtm": psnr_dtm,
        "lpips_chm": lpips_chm,
        "lpips_dtm": lpips_dtm,
        "mse": mse,
        "mae": mae,
    }


# ── Main sweep ────────────────────────────────────────────────────────────────
contorno = (ground_truth.sum(0) > 0).float().cpu().numpy()

for run_idx in range(1, NUM_RUNS + 1):
  # Determine which CSNRs still need processing for this run
  pending_csnrs = [
      csnr for csnr in CSNR_RANGE
      if not os.path.exists(
          os.path.join(base_result_path,
                       f"CR{COMPRESSION_RATIO}_CSNR{csnr}",
                       f"run{run_idx:02d}",
                       "metrics.json")
      ) or (
          RECOMPUTE_METRICS and os.path.exists(
              os.path.join(base_result_path,
                           f"CR{COMPRESSION_RATIO}_CSNR{csnr}",
                           f"run{run_idx:02d}",
                           "recon.npz")
          )
      )
  ]
  if not pending_csnrs:
      print(f"Run {run_idx}/{NUM_RUNS} already complete, skipping.")
      continue

  # New Poisson LiDAR realization for this run
  satellite_measurements, _ = forward_model(clean_gpu)
  satellite_measurements = satellite_measurements.cpu().unsqueeze(0).unsqueeze(0)

  for csnr in pending_csnrs:
    result_path = os.path.join(base_result_path, f"CR{COMPRESSION_RATIO}_CSNR{csnr}", f"run{run_idx:02d}")
    os.makedirs(result_path, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Run {run_idx}/{NUM_RUNS}  |  CSNR = {csnr} dB  |  CR = {COMPRESSION_RATIO}")
    print(f"Output: {result_path}")
    print(f"{'='*60}")

    # JSCC pass for this CSNR
    earth_signal = model(satellite_measurements, torch.tensor([float(csnr)])).detach().squeeze()
    earth_signal = earth_signal[:, :(96 // 3) * FACTOR, :(96 // 6) * FACTOR]
    input_data = earth_signal.to(DEVICE)
    print(f"Earth signal shape: {input_data.shape}")

    recon_path = os.path.join(result_path, 'recon.npz')
    if RECOMPUTE_METRICS and os.path.exists(recon_path):
        print(f"  Loading saved reconstruction for metric recomputation...")
        recon = np.load(recon_path)['recon']
    else:
        # Fresh noise initialization for each CSNR
        output = torch.randn_like(ground_truth.unsqueeze(0).float(), device=DEVICE)

        # Guided DDIM loop (DPS: Chung et al. 2022)
        pbar = tqdm(time_pairs, total=SAMPLING_TIMESTEPS, desc=f"run{run_idx:02d}/CSNR={csnr}")
        for time, time_next in pbar:
            output = output.detach().requires_grad_(True)

            time_cond = torch.full((1,), time, device='cuda', dtype=torch.long)
            pred_noise, x_start, *_ = grad_checkpoint(unet_wrapper, output, time_cond, use_reentrant=False)

            if time_next < 0:
                output_p = x_start.detach()
            else:
                alpha = trainer.ema.ema_model.alphas_cumprod[time]
                alpha_next = trainer.ema.ema_model.alphas_cumprod[time_next]

                sigma = ETA_DIFFUSION * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                c = (1 - alpha_next - sigma ** 2).sqrt()

                with torch.no_grad():
                    noise = torch.randn_like(output)
                    output_p = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

            x_start_scaled = ((x_start + 1) * 0.5) + EPSILON
            _, lambda_val = forward_model(x_start_scaled)
            nll = -Poisson(lambda_val).log_prob(input_data).mean()

            grads = torch.autograd.grad(nll, output, create_graph=False)[0]
            output = output_p - LR_MULTIPLIER * grads * (time / SAMPLING_TIMESTEPS)

            with torch.no_grad():
                x_start_scaled[x_start_scaled < RECON_THRESHOLD] = 0
                norm = (x_start_scaled - ground_truth.unsqueeze(0)).abs().mean()
            pbar.set_postfix(norm=norm.item(), lr=LR_MULTIPLIER, nll=nll.item(), t=time)
            del nll, grads, output_p, norm, lambda_val, x_start, x_start_scaled, pred_noise

            if time % 50 == 0:
                gc.collect()
                torch.cuda.empty_cache()

        # Post-processing
        output = ((output.detach() + 1) * 0.5)
        recon = output[0].cpu().numpy()
        recon[recon < RECON_THRESHOLD] = 0
        np.savez_compressed(recon_path, recon=recon)
        del output

    input_data_plot = input_data / (input_data.max(0)[0] + EPSILON)
    metrics = save_results(
        input_data_plot,
        recon * contorno,
        ground_truth,
        save_path=os.path.join(result_path, 'results.png')
    )

    metrics["csnr"] = csnr
    metrics["compression_ratio"] = COMPRESSION_RATIO
    metrics["run"] = run_idx
    with open(os.path.join(result_path, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"SSIM CHM: {metrics['ssim_chm']:.4f}, PSNR CHM: {metrics['psnr_chm']:.4f}")
    print(f"SSIM DTM: {metrics['ssim_dtm']:.4f}, PSNR DTM: {metrics['psnr_dtm']:.4f}")
    print(f"LPIPS CHM: {metrics['lpips_chm']:.4f}, LPIPS DTM: {metrics['lpips_dtm']:.4f}")
    print(f"MSE: {metrics['mse']:.4f}, MAE: {metrics['mae']:.4f}")

    # Free GPU memory before next CSNR
    del recon, input_data, earth_signal
    gc.collect()
    torch.cuda.empty_cache()

print("\nAll runs complete.")
