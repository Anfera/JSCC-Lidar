import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from forward_model import LidarForwardImagingModel
from canopyPlots import createCHM
from tqdm import tqdm
import matplotlib.pyplot as plt
import lpips
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
import os
import gc
import warnings
warnings.filterwarnings("ignore")

from torch.distributions import Poisson

# Constants
RESOLUTION = 2
ETA = 0.5
FACTOR = 3
SAMPLING_TIMESTEPS = 250
IMAGE_SIZE = 96 // RESOLUTION
DEVICE = 'cuda'
USE_OPTIMAL_CUBE = True
USE_AUTOENCODER_OUTPUT = True

LR_MULTIPLIER = 490366 if USE_OPTIMAL_CUBE else 5800
LR_MULTIPLIER = 50000 if USE_AUTOENCODER_OUTPUT else LR_MULTIPLIER

LR_MULTIPLIER = 4500
LR_MULTIPLIER = 10000

EPSILON = 1e-6
RECON_THRESHOLD = 0.05


def load_and_preprocess_data():
    """Load, crop, and normalize the ground truth voxel cube."""
    ground_truth = torch.from_numpy(np.swapaxes(np.load('TestCube/southeast.npy'), -1, -2)).float().to(DEVICE)
    ground_truth = ground_truth[:, :(96 // RESOLUTION) * FACTOR, :(96 // RESOLUTION) * FACTOR]
    ground_truth = ground_truth / (ground_truth.max(dim=0)[0] + EPSILON)
    ground_truth = torch.roll(ground_truth, shifts=20, dims=0)
    ground_truth = torch.flip(ground_truth, dims=[1, 2])
    return ground_truth


def plot_results(input_image: torch.Tensor,
                 sample: np.ndarray,
                 gt_image: torch.Tensor) -> None:
    """
    Plot CHM, DTM, and profile comparisons for input, reconstruction, and ground truth.

    Args:
        input_image (torch.Tensor): Input image tensor (forward model output, lower resolution).
        sample (np.ndarray): Reconstructed sample as numpy array.
        gt_image (torch.Tensor): Ground truth image tensor.
    """
    input_np = input_image.cpu().numpy()
    sample_np = sample
    gt_np = gt_image.cpu().numpy()

    chm_input, dtm_input, hillshade_input, dsm_input = createCHM(input_np, porcentaje=0.98)
    chm_recon, dtm_recon, hillshade_recon, dsm_recon = createCHM(sample_np, porcentaje=0.98)
    chm_gt, dtm_gt, hillshade_gt, dsm_gt = createCHM(gt_np, porcentaje=0.98)

    chm_input = chm_input * 0.5
    chm_recon = chm_recon * 0.5
    chm_gt = chm_gt * 0.5

    dtm_input = dtm_input * 0.5
    dtm_recon = dtm_recon * 0.5
    dtm_gt = dtm_gt * 0.5

    fig, axs = plt.subplots(3, 4, figsize=(15, 15))

    # CHM plots
    axs[0, 0].imshow(chm_input, cmap='viridis')
    axs[0, 0].set_aspect(0.5)
    axs[0, 0].set_title('CHM Input')

    axs[0, 1].imshow(chm_recon, cmap='viridis', vmin=chm_gt.min(), vmax=chm_gt.max())

    axs[0, 1].set_title('CHM Reconstruction')

    axs[0, 2].imshow(chm_gt, cmap='viridis')
    axs[0, 2].set_title('CHM Ground Truth')

    # DTM plots with hillshade overlay
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

    # Profile plots — input may have different spatial resolution than recon/GT
    profile_index_input = 3 * input_np.shape[1] // 4
    profile_index_gt = 3 * gt_np.shape[1] // 4

    axs[2, 0].imshow(input_np[::-1, profile_index_input, :], cmap='gray_r', interpolation='nearest')
    axs[2, 0].set_aspect(1 / 3)
    axs[2, 0].set_title('Profile Input')

    axs[2, 1].imshow(sample_np[::-1, profile_index_gt, :], cmap='gray_r', interpolation='nearest')
    axs[2, 1].set_title('Profile Reconstruction')

    axs[2, 2].imshow(gt_np[::-1, profile_index_gt, :], cmap='gray_r', interpolation='nearest')
    axs[2, 2].set_title('Profile Ground Truth')

    # Error plots
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

    plt.show()

    # Metrics
    ssim_chm = ssim(dsm_gt, dsm_recon, data_range=dsm_gt.max() - dsm_gt.min(), win_size=11)
    psnr_chm = psnr(dsm_gt, dsm_recon, data_range=dsm_gt.max() - dsm_gt.min())
    ssim_dtm = ssim(dtm_gt, dtm_recon, data_range=dtm_gt.max() - dtm_gt.min(), win_size=11)
    psnr_dtm = psnr(dtm_gt, dtm_recon, data_range=dtm_gt.max() - dtm_gt.min())
    print(f"SSIM CHM: {ssim_chm:.4f}, PSNR CHM: {psnr_chm:.4f}")
    print(f"SSIM DTM: {ssim_dtm:.4f}, PSNR DTM: {psnr_dtm:.4f}")

    lpips_fn = lpips.LPIPS(net='vgg')
    lpips_chm = lpips_fn.forward(torch.tensor(dsm_gt).float().unsqueeze(0).unsqueeze(0),
                                  torch.tensor(dsm_recon).float().unsqueeze(0).unsqueeze(0)).item()
    lpips_dtm = lpips_fn.forward(torch.tensor(dtm_gt).float().unsqueeze(0).unsqueeze(0),
                                  torch.tensor(dtm_recon).float().unsqueeze(0).unsqueeze(0)).item()
    print(f"LPIPS CHM: {lpips_chm:.4f}, LPIPS DTM: {lpips_dtm:.4f}")

    mse = torch.mean((torch.tensor(gt_np) - torch.tensor(sample_np)) ** 2).item()
    mae = torch.mean(torch.abs(torch.tensor(gt_np) - torch.tensor(sample_np))).item()
    print(f"MSE: {mse:.4f}, MAE: {mae:.4f}")


def main():
    """Main function to execute the diffusion model inference and visualization."""

    # Initialize forward imaging model
    forward_imaging = LidarForwardImagingModel(
        output_res_m=(3.0, 6.0),
        footprint_diameter_m=10.0,
        b=0.1,    # background photon rate
        eta=ETA,  # Gaussian noise standard deviation
        ref_altitude=500.0,
        ref_photon_count=20.0,
    )
    forward_imaging = forward_imaging.to(DEVICE)

    # Load and preprocess data
    ground_truth = load_and_preprocess_data()
    # input_data = forward_imaging(ground_truth)[0]
    
    # if USE_OPTIMAL_CUBE:
    #     input_data = forward_imaging(ground_truth)[1][0]

    if USE_AUTOENCODER_OUTPUT:
        input_data = np.load('TestCube/reconstructionFL.npy')[:, :(96 // 3) * FACTOR, :(96 // 6) * FACTOR]
        input_data = torch.from_numpy(input_data).float().to(DEVICE)
        

    # Define U-Net and diffusion model
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

    # Initialize Trainer and load pre-trained weights
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

    # DDIM schedule: eta=1.0 gives DDPM-equivalent stochasticity
    times = torch.linspace(-1, 1000 - 1, steps=SAMPLING_TIMESTEPS + 1)  # [-1, 0, 1, ..., T-1]
    times = list(reversed(times.int().tolist()))
    time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), ..., (1, 0), (0, -1)]
    eta = 1.0 # do not confuse with forward model's eta; this is the DDIM noise scale

    # Initialize from pure noise
    output = torch.randn_like(ground_truth.unsqueeze(0).float(), device=DEVICE)

    # Guided DDIM loop (DPS: Chung et al. 2022)
    pbar = tqdm(time_pairs, total=SAMPLING_TIMESTEPS)
    for time, time_next in pbar:
        output = output.detach().requires_grad_(True)

        time_cond = torch.full((1,), time, device='cuda', dtype=torch.long)
        pred_noise, x_start, *_ = checkpoint(unet_wrapper, output, time_cond, use_reentrant=False)

        if time_next < 0:
            # Final step: x_start is the clean estimate; no DDIM transition needed
            output_p = x_start.detach()
        else:
            alpha = trainer.ema.ema_model.alphas_cumprod[time]
            alpha_next = trainer.ema.ema_model.alphas_cumprod[time_next]

            # DDIM sigma (Song et al. 2020, eq. 12)
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            with torch.no_grad():
                noise = torch.randn_like(output)
                output_p = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

        # DPS guidance: gradient of the NLL w.r.t. x_t to enforce data consistency
        x_start_scaled = ((x_start + 1) * 0.5) + EPSILON
        _, lambda_val = forward_imaging(x_start_scaled)
        var = lambda_val + forward_imaging.eta**2 + EPSILON
        res = input_data - lambda_val
        nll = -Poisson(lambda_val).log_prob(input_data).mean() if USE_OPTIMAL_CUBE else 0.5 * ((res**2) / var + torch.log(var)).mean()
        # nll = (res).abs().mean()
        # nll = torch.nn.functional.kl_div(lambda_val.log(),input_data, reduction='none').sum(0).mean()

        grads = torch.autograd.grad(nll, output, create_graph=False)[0]

        output = output_p - LR_MULTIPLIER * grads * (time/SAMPLING_TIMESTEPS)

        with torch.no_grad():
            x_start_scaled[x_start_scaled < RECON_THRESHOLD] = 0
            norm = (x_start_scaled - ground_truth.unsqueeze(0)).abs().mean()
        pbar.set_postfix(norm=norm.item(), lr=LR_MULTIPLIER, log_likelihood=nll.item(), time=time)
        del nll, grads, output_p, norm, lambda_val, var, res, x_start, x_start_scaled, pred_noise

        if time % 50 == 0:
            gc.collect()
            torch.cuda.empty_cache()
            alloc = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"[t={time}] allocated={alloc:.2f} GB, reserved={reserved:.2f} GB")

    # Post-process: map from [-1, 1] diffusion space to [0, 1]
    output = ((output.detach() + 1) * 0.5)
    recon = output[0].cpu().numpy()
    recon[recon < RECON_THRESHOLD] = 0

    # np.save(os.path.join(result_path, f"reconstruction_{USE_OPTIMAL_CUBE}_{USE_AUTOENCODER_OUTPUT}.npy"), recon)

    contorno = (ground_truth.sum(0) > 0).float().cpu().numpy()
    # input_data = input_data / (input_data.max(0)[0] + EPSILON)
    plot_results(input_data - 0.15, recon * contorno, ground_truth)

if __name__ == "__main__":
    main()