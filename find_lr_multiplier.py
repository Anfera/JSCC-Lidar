import csv
import gc
import warnings
warnings.filterwarnings("ignore")

import lpips as lpips_lib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.distributions import Poisson
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from forward_model import LidarForwardImagingModel
from canopyPlots import createCHM
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

# Reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Constants
RESOLUTION = 2
FACTOR = 3
SAMPLING_TIMESTEPS = 250
IMAGE_SIZE = 96 // RESOLUTION
DEVICE = 'cuda'
EPSILON = 1e-6
RECON_THRESHOLD = 0.05
SCENARIOS = [
    # {'name': 'noisy',       'use_poisson': False, 'use_autoencoder': False},
    # {'name': 'oracle',      'use_poisson': True,  'use_autoencoder': False},
    {'name': 'autoencoder', 'use_poisson': True,  'use_autoencoder': True},
]

# ETA_VALUES = [0.1, 0.5, 1.0, 2.0]
ETA_VALUES = [0.5]
LR_COARSE_VALUES = np.logspace(3, 6, 15).tolist()  # 15 values log-spaced
LR_FINE_STEPS = 5  # number of interior points in the fine search

# Set to 'psnr_chm' to reproduce original behavior, or 'composite' to optimize all metrics
SEARCH_METRIC = 'composite'

# Weights for composite score [psnr_chm, psnr_dtm, ssim_chm, ssim_dtm, lpips_chm, lpips_dtm, psnr_tensor, ssim_tensor, lpips_tensor]
# Only used when SEARCH_METRIC = 'composite'
METRIC_WEIGHTS = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]  # equal weights


def get_fine_lr_values(coarse_norms, n_steps=LR_FINE_STEPS):
    sorted_lrs = sorted(coarse_norms, key=lambda x: x[0])
    best_lr = max(coarse_norms, key=lambda x: x[1])[0]
    lrs_only = [x[0] for x in sorted_lrs]
    best_idx = lrs_only.index(best_lr)
    lo = lrs_only[max(0, best_idx - 1)]
    hi = lrs_only[min(len(lrs_only) - 1, best_idx + 1)]
    return np.linspace(lo, hi, n_steps + 2)[1:-1].tolist()  # exclude endpoints (already sampled)


def compute_composite_scores(raw_results):
    """raw_results: [(lr, metrics_dict), ...] -> [(lr, composite_score), ...]"""
    keys_pos = ['psnr_chm', 'psnr_dtm', 'ssim_chm', 'ssim_dtm', 'psnr_tensor', 'ssim_tensor']  # higher is better
    keys_neg = ['lpips_chm', 'lpips_dtm', 'lpips_tensor']                                       # lower is better
    w = METRIC_WEIGHTS
    n = len(raw_results)
    scores = np.zeros(n)
    for i, key in enumerate(keys_pos):
        vals = np.array([m[key] for _, m in raw_results])
        vmin, vmax = vals.min(), vals.max()
        if vmax > vmin:
            scores += w[i] * (vals - vmin) / (vmax - vmin)
    for j, key in enumerate(keys_neg):
        vals = np.array([m[key] for _, m in raw_results])
        vmin, vmax = vals.min(), vals.max()
        if vmax > vmin:
            scores += w[len(keys_pos) + j] * (1.0 - (vals - vmin) / (vmax - vmin))
    total_w = sum(METRIC_WEIGHTS)
    scores /= total_w
    return [(raw_results[i][0], float(scores[i])) for i in range(n)]


def get_search_scores(raw_results):
    """Convert raw_results to [(lr, scalar), ...] for use in get_fine_lr_values."""
    if SEARCH_METRIC == 'psnr_chm':
        return [(lr, m['psnr_chm']) for lr, m in raw_results]
    return compute_composite_scores(raw_results)


def load_and_preprocess_data():
    """Load, crop, and normalize the ground truth voxel cube."""
    ground_truth = torch.from_numpy(np.swapaxes(np.load('TestCube/southeast.npy'), -1, -2)).float().to(DEVICE)
    ground_truth = ground_truth[:, :(96 // RESOLUTION) * FACTOR, :(96 // RESOLUTION) * FACTOR]
    ground_truth = ground_truth / (ground_truth.max(dim=0)[0] + EPSILON)
    ground_truth = torch.roll(ground_truth, shifts=20, dims=0)
    ground_truth = torch.flip(ground_truth, dims=[1, 2])
    return ground_truth


def get_input_data(scenario, forward_imaging, ground_truth):
    if scenario['use_autoencoder']:
        data = np.load('TestCube/reconstructionFL.npy')[:, :(96 // 3) * FACTOR, :(96 // 6) * FACTOR]
        return torch.from_numpy(data).float().to(DEVICE)
    with torch.no_grad():
        idx = 1 if scenario['use_poisson'] else 0
        return forward_imaging(ground_truth)[idx].squeeze()


def run_inference(eta_val, lr_multiplier, forward_imaging, input_data, ground_truth, trainer, lpips_fn, use_poisson=True, return_output=False):
    """Run DDIM+DPS inference and return a dict of all quality metrics on CHM and DTM."""

    def unet_wrapper(x, time_cond):
        return trainer.ema.ema_model.model_predictions(
            x, time_cond, x_self_cond=None, clip_x_start=True, rederive_pred_noise=True
        )

    times = torch.linspace(-1, 1000 - 1, steps=SAMPLING_TIMESTEPS + 1)
    times = list(reversed(times.int().tolist()))
    time_pairs = list(zip(times[:-1], times[1:]))
    ddim_eta = 1.0  # DDIM noise scale (not the forward model's eta)

    output = torch.randn_like(ground_truth.unsqueeze(0).float(), device=DEVICE)

    for time, time_next in tqdm(time_pairs, total=SAMPLING_TIMESTEPS,
                                desc=f"ETA={eta_val} LR={lr_multiplier:.0f}",
                                leave=False):
        output = output.detach().requires_grad_(True)

        time_cond = torch.full((1,), time, device=DEVICE, dtype=torch.long)
        pred_noise, x_start, *_ = checkpoint(unet_wrapper, output, time_cond, use_reentrant=False)

        if time_next < 0:
            output_p = x_start.detach()
        else:
            alpha = trainer.ema.ema_model.alphas_cumprod[time]
            alpha_next = trainer.ema.ema_model.alphas_cumprod[time_next]
            sigma = ddim_eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            with torch.no_grad():
                noise = torch.randn_like(output)
                output_p = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

        # DPS guidance
        x_start_scaled = ((x_start + 1) * 0.5) + EPSILON
        _, lambda_val = forward_imaging(x_start_scaled)
        var = lambda_val + forward_imaging.eta ** 2 + EPSILON
        res = input_data - lambda_val
        nll = -Poisson(lambda_val).log_prob(input_data).mean() if use_poisson else 0.5 * ((res ** 2) / var + torch.log(var)).mean()

        grads = torch.autograd.grad(nll, output, create_graph=False)[0]
        output = output_p - lr_multiplier * grads * (time / SAMPLING_TIMESTEPS)

        del nll, grads, output_p, lambda_val, var, res, x_start, x_start_scaled, pred_noise

        if time % 50 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    # Map from [-1, 1] diffusion space to [0, 1] and threshold
    output = ((output.detach() + 1) * 0.5)
    output[output < RECON_THRESHOLD] = 0

    output_np = output[0].cpu().numpy()
    gt_np = ground_truth.cpu().numpy()

    _, dtm_recon, _, chm_recon = createCHM(output_np, porcentaje=0.95)
    _, dtm_gt, _, chm_gt = createCHM(gt_np, porcentaje=0.95)
    chm_recon = chm_recon * 0.5
    chm_gt = chm_gt * 0.5
    dtm_recon = dtm_recon * 0.5
    dtm_gt = dtm_gt * 0.5

    psnr_chm = psnr(chm_gt, chm_recon, data_range=chm_gt.max() - chm_gt.min())
    psnr_dtm = psnr(dtm_gt, dtm_recon, data_range=dtm_gt.max() - dtm_gt.min())
    ssim_chm = ssim(chm_gt, chm_recon, data_range=chm_gt.max() - chm_gt.min(), win_size=13)
    ssim_dtm = ssim(dtm_gt, dtm_recon, data_range=dtm_gt.max() - dtm_gt.min(), win_size=13)

    with torch.no_grad():
        lpips_chm = lpips_fn(
            torch.tensor(chm_gt).float().unsqueeze(0).unsqueeze(0).to(DEVICE),
            torch.tensor(chm_recon).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
        ).item()
        lpips_dtm = lpips_fn(
            torch.tensor(dtm_gt).float().unsqueeze(0).unsqueeze(0).to(DEVICE),
            torch.tensor(dtm_recon).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
        ).item()

    # Whole-tensor metrics
    data_range_tensor = gt_np.max() - gt_np.min()
    psnr_tensor = psnr(gt_np, output_np, data_range=data_range_tensor)
    ssim_tensor = ssim(gt_np, output_np, data_range=data_range_tensor, channel_axis=0, win_size=11)
    lpips_slices = []
    with torch.no_grad():
        for z in range(gt_np.shape[2]):
            lpips_slices.append(lpips_fn(
                torch.tensor(gt_np[:, :, z]).float().unsqueeze(0).unsqueeze(0).to(DEVICE),
                torch.tensor(output_np[:, :, z]).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
            ).item())
    lpips_tensor = float(np.mean(lpips_slices))

    del output
    gc.collect()
    torch.cuda.empty_cache()

    metrics = {
        'psnr_chm': psnr_chm,
        'psnr_dtm': psnr_dtm,
        'ssim_chm': ssim_chm,
        'ssim_dtm': ssim_dtm,
        'lpips_chm': lpips_chm,
        'lpips_dtm': lpips_dtm,
        'psnr_tensor': psnr_tensor,
        'ssim_tensor': ssim_tensor,
        'lpips_tensor': lpips_tensor,
    }
    if return_output:
        return metrics, output_np
    return metrics


def plot_results(input_image, sample, gt_image, lpips_fn, title, save_path):
    """Plot CHM, DTM, and profile comparisons for input, reconstruction, and ground truth."""
    input_np = input_image.cpu().numpy() if isinstance(input_image, torch.Tensor) else input_image
    sample_np = sample
    gt_np = gt_image.cpu().numpy() if isinstance(gt_image, torch.Tensor) else gt_image

    chm_input, dtm_input, hillshade_input, _ = createCHM(input_np, porcentaje=0.98)
    chm_recon, dtm_recon, hillshade_recon, _ = createCHM(sample_np, porcentaje=0.98)
    chm_gt, dtm_gt, hillshade_gt, _ = createCHM(gt_np, porcentaje=0.98)

    chm_input = chm_input * 0.5
    chm_recon = chm_recon * 0.5
    chm_gt    = chm_gt    * 0.5
    dtm_input = dtm_input * 0.5
    dtm_recon = dtm_recon * 0.5
    dtm_gt    = dtm_gt    * 0.5

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
    profile_index_gt    = 3 * gt_np.shape[1]    // 4

    axs[2, 0].imshow(input_np[::-1, profile_index_input, :], cmap='gray_r', interpolation='nearest')
    axs[2, 0].set_aspect(1 / 3)
    axs[2, 0].set_title('Profile Input')
    axs[2, 1].imshow(sample_np[::-1, profile_index_gt, :], cmap='gray_r', interpolation='nearest')
    axs[2, 1].set_title('Profile Reconstruction')
    axs[2, 2].imshow(gt_np[::-1, profile_index_gt, :], cmap='gray_r', interpolation='nearest')
    axs[2, 2].set_title('Profile Ground Truth')

    error_chm     = np.abs(chm_recon - chm_gt)
    error_dtm     = np.abs(dtm_recon - dtm_gt)
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

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved {save_path}")

    ssim_chm = ssim(chm_gt, chm_recon, data_range=chm_gt.max() - chm_gt.min(), win_size=11)
    psnr_chm = psnr(chm_gt, chm_recon, data_range=chm_gt.max() - chm_gt.min())
    ssim_dtm = ssim(dtm_gt, dtm_recon, data_range=dtm_gt.max() - dtm_gt.min(), win_size=11)
    psnr_dtm = psnr(dtm_gt, dtm_recon, data_range=dtm_gt.max() - dtm_gt.min())
    print(f"  SSIM CHM: {ssim_chm:.4f}, PSNR CHM: {psnr_chm:.4f}")
    print(f"  SSIM DTM: {ssim_dtm:.4f}, PSNR DTM: {psnr_dtm:.4f}")

    with torch.no_grad():
        lpips_chm = lpips_fn(
            torch.tensor(chm_gt).float().unsqueeze(0).unsqueeze(0).to(DEVICE),
            torch.tensor(chm_recon).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
        ).item()
        lpips_dtm = lpips_fn(
            torch.tensor(dtm_gt).float().unsqueeze(0).unsqueeze(0).to(DEVICE),
            torch.tensor(dtm_recon).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
        ).item()
    print(f"  LPIPS CHM: {lpips_chm:.4f}, LPIPS DTM: {lpips_dtm:.4f}")

    mse = np.mean((gt_np - sample_np) ** 2)
    mae = np.mean(np.abs(gt_np - sample_np))
    print(f"  MSE: {mse:.4f}, MAE: {mae:.4f}")


def main():
    ground_truth = load_and_preprocess_data()

    model = Unet(
        dim=128,
        dim_mults=(8, 16, 16, 16),
        flash_attn=True,
        channels=128
    )
    diffusion = GaussianDiffusion(
        model,
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

    lpips_fn = lpips_lib.LPIPS(net='vgg').to(DEVICE)

    all_results = {}  # {(scenario_name, eta_val): [(lr, metrics_dict), ...]}

    for eta_val in ETA_VALUES:
        forward_imaging = LidarForwardImagingModel(
            output_res_m=(3.0, 6.0),
            footprint_diameter_m=10.0,
            b=0.1,
            eta=eta_val,
            ref_altitude=500.0,
            ref_photon_count=20.0,
        ).to(DEVICE)

        for scenario in SCENARIOS:
            sname = scenario['name']
            print(f"\n{'='*65}")
            print(f"  Searching LR_MULTIPLIER for ETA = {eta_val}  input = {sname}  [mode: {SEARCH_METRIC}]")
            print(f"{'='*65}")

            input_data = get_input_data(scenario, forward_imaging, ground_truth)

            raw_results = []
            for lr in LR_COARSE_VALUES:
                m = run_inference(eta_val, lr, forward_imaging, input_data, ground_truth, trainer, lpips_fn,
                                  use_poisson=scenario['use_poisson'])
                raw_results.append((lr, m))
                print(f"  [{sname}] ETA={eta_val}  LR={lr:9.1f}  psnr_chm={m['psnr_chm']:.4f}"
                      f"  ssim_chm={m['ssim_chm']:.4f}  lpips_chm={m['lpips_chm']:.4f}"
                      f"  psnr_dtm={m['psnr_dtm']:.4f}  ssim_dtm={m['ssim_dtm']:.4f}"
                      f"  lpips_dtm={m['lpips_dtm']:.4f}"
                      f"  psnr_tensor={m['psnr_tensor']:.4f}  ssim_tensor={m['ssim_tensor']:.4f}"
                      f"  lpips_tensor={m['lpips_tensor']:.4f}")

            scored_coarse = get_search_scores(raw_results)
            fine_lrs = get_fine_lr_values(scored_coarse)
            print(f"\n  Fine search in [{fine_lrs[0]:.1f}, {fine_lrs[-1]:.1f}] ({len(fine_lrs)} pts)")
            for lr in fine_lrs:
                m = run_inference(eta_val, lr, forward_imaging, input_data, ground_truth, trainer, lpips_fn,
                                  use_poisson=scenario['use_poisson'])
                raw_results.append((lr, m))
                print(f"  [fine/{sname}] ETA={eta_val}  LR={lr:9.1f}  psnr_chm={m['psnr_chm']:.4f}"
                      f"  ssim_chm={m['ssim_chm']:.4f}  lpips_chm={m['lpips_chm']:.4f}"
                      f"  psnr_dtm={m['psnr_dtm']:.4f}  ssim_dtm={m['ssim_dtm']:.4f}"
                      f"  lpips_dtm={m['lpips_dtm']:.4f}"
                      f"  psnr_tensor={m['psnr_tensor']:.4f}  ssim_tensor={m['ssim_tensor']:.4f}"
                      f"  lpips_tensor={m['lpips_tensor']:.4f}")

            all_results[(sname, eta_val)] = raw_results
            scored_all = get_search_scores(raw_results)
            best_lr, best_score = max(scored_all, key=lambda x: x[1])
            best_m = next(m for lr, m in raw_results if lr == best_lr)
            print(f"  >>> Best LR for [{sname}] ETA={eta_val}: {best_lr:.1f}  (score={best_score:.5f},"
                  f" psnr_chm={best_m['psnr_chm']:.4f}, ssim_chm={best_m['ssim_chm']:.4f},"
                  f" lpips_chm={best_m['lpips_chm']:.4f})")

    # --- Save CSV ---
    with open('lr_search_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['eta', 'input_type', 'lr_multiplier', 'psnr_chm', 'ssim_chm', 'lpips_chm',
                         'psnr_dtm', 'ssim_dtm', 'lpips_dtm',
                         'psnr_tensor', 'ssim_tensor', 'lpips_tensor', 'composite'])
        for (sname, eta_val), raw_results in all_results.items():
            scored = compute_composite_scores(raw_results)
            score_map = {lr: s for lr, s in scored}
            for lr, m in raw_results:
                writer.writerow([
                    eta_val, sname, f"{lr:.4f}",
                    f"{m['psnr_chm']:.6f}", f"{m['ssim_chm']:.6f}", f"{m['lpips_chm']:.6f}",
                    f"{m['psnr_dtm']:.6f}", f"{m['ssim_dtm']:.6f}", f"{m['lpips_dtm']:.6f}",
                    f"{m['psnr_tensor']:.6f}", f"{m['ssim_tensor']:.6f}", f"{m['lpips_tensor']:.6f}",
                    f"{score_map[lr]:.6f}",
                ])
    print("\nSaved lr_search_results.csv")

    # --- Summary table ---
    print("\n--- Optimal LR_MULTIPLIER per input type and ETA ---")
    print(f"{'Input':>12} | {'ETA':>6} | {'Best LR':>12} | {'Score':>8} | {'PSNR CHM':>10} | {'SSIM CHM':>10} | {'LPIPS CHM':>10}")
    print("-" * 84)
    for (sname, eta_val), raw_results in all_results.items():
        scored = get_search_scores(raw_results)
        best_lr, best_score = max(scored, key=lambda x: x[1])
        best_m = next(m for lr, m in raw_results if lr == best_lr)
        print(f"{sname:>12} | {eta_val:>6.2f} | {best_lr:>12.1f} | {best_score:>8.5f}"
              f" | {best_m['psnr_chm']:>10.4f} | {best_m['ssim_chm']:>10.4f}"
              f" | {best_m['lpips_chm']:>10.4f}")

    # --- Plot ---
    metric_keys = ['psnr_chm', 'ssim_chm', 'lpips_chm', 'psnr_dtm', 'ssim_dtm', 'lpips_dtm',
                   'psnr_tensor', 'ssim_tensor', 'lpips_tensor']
    metric_labels = ['PSNR CHM (dB)', 'SSIM CHM', 'LPIPS CHM', 'PSNR DTM (dB)', 'SSIM DTM', 'LPIPS DTM',
                     'PSNR Tensor (dB)', 'SSIM Tensor', 'LPIPS Tensor']
    fig, axes = plt.subplots(3, 3, figsize=(16, 13))
    axes = axes.flatten()

    for ax, key, label in zip(axes, metric_keys, metric_labels):
        for (sname, eta_val), raw_results in all_results.items():
            lrs = [lr for lr, _ in raw_results]
            vals = [m[key] for _, m in raw_results]
            scored = get_search_scores(raw_results)
            best_lr, _ = max(scored, key=lambda x: x[1])
            best_val = next(m[key] for lr, m in raw_results if lr == best_lr)
            line, = ax.plot(lrs, vals, marker='o', label=f'ETA={eta_val} [{sname}]')
            ax.plot(best_lr, best_val, marker='*', markersize=12, color=line.get_color(), zorder=5)
        ax.set_xscale('log')
        ax.set_xlabel('LR_MULTIPLIER')
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend()
        ax.grid(True, which='both', alpha=0.3)

    fig.suptitle(f'LR_MULTIPLIER Search  (FACTOR={FACTOR}, T={SAMPLING_TIMESTEPS}, mode={SEARCH_METRIC})',
                 fontsize=13)
    plt.tight_layout()
    plt.savefig('lr_search_results.png', dpi=150)
    plt.close(fig)
    print("Saved lr_search_results.png")

    # --- Reconstruction plots for optimal parameters ---
    print("\n--- Generating reconstruction plots for optimal LR_MULTIPLIER per config ---")
    for eta_val in ETA_VALUES:
        forward_imaging = LidarForwardImagingModel(
            output_res_m=(3.0, 6.0),
            footprint_diameter_m=10.0,
            b=0.1,
            eta=eta_val,
            ref_altitude=500.0,
            ref_photon_count=20.0,
        ).to(DEVICE)

        for scenario in SCENARIOS:
            sname = scenario['name']
            raw_results = all_results[(sname, eta_val)]
            scored = get_search_scores(raw_results)
            best_lr, best_score = max(scored, key=lambda x: x[1])

            print(f"\n  Running final inference: [{sname}] ETA={eta_val}  LR={best_lr:.1f}")
            input_data = get_input_data(scenario, forward_imaging, ground_truth)
            _, output_np = run_inference(
                eta_val, best_lr, forward_imaging, input_data, ground_truth,
                trainer, lpips_fn, use_poisson=scenario['use_poisson'],
                return_output=True
            )

            contorno = (ground_truth.sum(0) > 0).float().cpu().numpy()
            output_np = output_np * contorno

            input_plot = input_data / (input_data.max(0)[0] + EPSILON)

            title = (f'Reconstruction — {sname}  ETA={eta_val}  LR={best_lr:.1f}'
                     f'  (score={best_score:.5f})')
            save_path = f'recon_{sname}_eta{eta_val:.1f}.png'
            plot_results(input_plot, output_np, ground_truth, lpips_fn, title, save_path)


if __name__ == "__main__":
    main()
