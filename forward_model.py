import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LidarForwardImagingModel(nn.Module):
    def __init__(
        self,
        input_res_m=(2.0, 2.0),
        output_res_m=(3.0, 6.0),
        footprint_diameter_m=10.0,
        b=0.1,
        eta=0.5,
        ref_altitude=500.0,
        ref_photon_count=20.0,
    ):
        """
        Args:
            input_res_m (tuple): Physical size of input pixels (dy, dx) in meters.
            output_res_m (tuple): Physical size of output pixels (dy, dx) in meters.
            footprint_diameter_m (float): The 1/e^2 beam diameter in meters.
            b (float): Background noise.
            eta (float): Readout noise.
            ref_altitude (float): Reference altitude (km).
            ref_photon_count (float): Target photon count.
        """
        super().__init__()
        self.b = b
        self.eta = eta
        self.ref_altitude = ref_altitude
        self.ref_photon_count = ref_photon_count

        self.input_res_m = input_res_m
        self.output_res_m = output_res_m

        # 1. Calculate Area Scale Factor
        in_area = input_res_m[0] * input_res_m[1]
        out_area = output_res_m[0] * output_res_m[1]
        self.area_scale_factor = out_area / in_area

        # 2. Calculate Sigma in Input Pixels
        # Def: 1/e^2 diameter is 4 * sigma
        sigma_m = footprint_diameter_m / 4.0

        avg_input_res = (input_res_m[0] + input_res_m[1]) / 2.0
        sigma_px = sigma_m / avg_input_res

        # 3. Create Base Kernel
        # We use 6*sigma for the kernel size to capture >99% of energy
        # (Since diameter is 4*sigma, this means kernel is 1.5x the footprint size)
        kernel_size = int(math.ceil(6 * sigma_px))
        if kernel_size % 2 == 0:
            kernel_size += 1

        self.register_buffer("kernel", self._create_gaussian_kernel(kernel_size, sigma_px))

        print(f"Model Initialized: In {input_res_m}m -> Out {output_res_m}m")
        print(f"Footprint (1/e^2): {footprint_diameter_m}m (Sigma: {sigma_m:.2f}m / {sigma_px:.2f} px)")

    def _create_gaussian_kernel(self, size, sigma):
        coords = torch.arange(size).float() - (size - 1) / 2
        x_grid, y_grid = torch.meshgrid(coords, coords, indexing='ij')
        kernel = torch.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, size, size)

    def forward(self, X_h, altitude=500.0):
        if X_h.ndim == 3:
            X_h = X_h.unsqueeze(0)

        batch_size, num_bins, h_in, w_in = X_h.shape

        # --- 1. Dynamic Output Size ---
        fov_h_m = h_in * self.input_res_m[0]
        fov_w_m = w_in * self.input_res_m[1]

        out_h = int(fov_h_m / self.output_res_m[0])
        out_w = int(fov_w_m / self.output_res_m[1])
        output_size = (out_h, out_w)

        # --- 2. Physics Normalization ---
        energy_per_tube = X_h.sum(dim=1, keepdim=True)
        global_mean_energy = energy_per_tube.mean(dim=(2, 3), keepdim=True)
        X_norm = X_h / (global_mean_energy + 1e-8)

        dist_scale = (self.ref_altitude / altitude) ** 2
        target_intensity = (self.ref_photon_count / self.area_scale_factor) * dist_scale
        X_scaled = X_norm * target_intensity

        # --- 3. Spatial Blurring ---
        # Expand single-channel kernel to match height bins
        current_kernel = self.kernel.repeat(num_bins, 1, 1, 1)

        padding = current_kernel.shape[-1] // 2
        X_blurred = F.conv2d(X_scaled, current_kernel, padding=padding, groups=num_bins)

        # --- 4. Downsampling ---
        X_binned = F.interpolate(X_blurred, size=output_size, mode='area')
        X_integrated = X_binned * self.area_scale_factor

        # --- 5. Noise ---
        lambda_val = torch.relu(X_integrated) + self.b
        X_l = torch.poisson(lambda_val)
        gaussian_noise = torch.randn_like(X_l) * self.eta
        Y_l = X_l + gaussian_noise

        if Y_l.shape[0] == 1:
            Y_l = Y_l.squeeze(0)

        return Y_l, lambda_val


def estimate_b_eta(Y, patch_size=8, step=None, retain_fraction=0.1):
    """
    Robustly estimate b and eta from Y ~ Poisson(S + b) + N(0, eta^2)
    Filters out structural variance by only fitting the 'flattest' patches.
    """
    if isinstance(Y, torch.Tensor):
        Y = Y.detach().cpu().numpy()
    
    # Ensure shape is at least 3D (N, H, W) to process images independently
    Y = np.atleast_3d(Y) 
    if Y.ndim == 4:
        Y = Y.reshape(-1, Y.shape[-2], Y.shape[-1])
        
    N_imgs, h, w = Y.shape
    step = step or max(1, patch_size // 2)

    means, vars_list = [], []

    # 1. Extract patches independently per 2D slice
    for n in range(N_imgs):
        img = Y[n]
        for i in range(0, h - patch_size + 1, step):
            for j in range(0, w - patch_size + 1, step):
                patch = img[i:i+patch_size, j:j+patch_size]
                means.append(float(np.mean(patch)))
                vars_list.append(float(np.var(patch, ddof=1)))

    means = np.array(means)
    vars_ = np.array(vars_list)

    if len(means) < 20:
        print("Warning: Not enough patches for robust estimation.")
        return 0.0, 0.0

    # 2. Estimate b (Assuming the darkest patches have zero target signal)
    b_est = max(0.0, np.percentile(means, 3.0))

    # 3. Filter for 'flat' patches to estimate eta
    # Theoretical flat patch: Var = Mean + eta^2 -> Var - Mean = eta^2
    # Patches with structure will have: Var >> Mean + eta^2
    # So we sort by (Var - Mean) and keep the lowest fraction.
    diff = vars_ - means
    
    num_to_keep = max(10, int(len(means) * retain_fraction))
    flat_indices = np.argsort(diff)[:num_to_keep]
    
    flat_means = means[flat_indices]
    flat_vars = vars_[flat_indices]

    # 4. Linear fit on the flat patches
    A = np.vstack([flat_means, np.ones_like(flat_means)]).T
    slope, intercept = np.linalg.lstsq(A, flat_vars, rcond=None)[0]

    eta_est = np.sqrt(max(0.0, intercept))

    print(f"Robust Est → b={b_est:.4f}, eta={eta_est:.4f} (Fit slope={slope:.3f} on {num_to_keep} patches)")
    return b_est, eta_est

if __name__ == "__main__":
    
    # Example usage
    model = LidarForwardImagingModel(
        input_res_m=(2.0, 2.0),
        output_res_m=(3.0, 6.0),
        footprint_diameter_m=10.0,
        b=0.1,
        eta=0.5,
        ref_altitude=500.0,
        ref_photon_count=20.0,
    )
    dummy_input = torch.rand(1, 128, 48, 48)  # (B, C, H_in, W_in)
    output, lambda_val = model(dummy_input)
    output = output.squeeze(0)
    print("Output shape:", output.shape)
    casals = torch.from_numpy(np.load('TestCube/hhdc_casals_resampled.npy')[:,:,:512])
    print("Loaded Casals data shape:", casals.shape)

    # output = 5*(output.numpy() / (np.max(output.numpy(), axis=0) + 1e-8))
    # casals = 5*(casals / (casals.max(0)[0]+1e-8))

    print(output.max(), casals.max())

    b_st, eta_est = estimate_b_eta(output)
    b_st, eta_est = estimate_b_eta(casals)