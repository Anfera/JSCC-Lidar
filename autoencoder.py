import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

# =================================================================================
# Helper Modules
# =================================================================================

class ResidualBlock(nn.Module):
    """
    A standard residual block with two 3D convolutional layers.
    Uses GroupNorm for normalization, which is batch-size independent.
    *FIX APPLIED: Added padding_mode='replicate' to prevent boundary zero-drop*
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode='replicate')
        self.norm1 = nn.GroupNorm(8, out_channels) # 8 groups, a common choice
        self.act1 = nn.SiLU() # Swish activation
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode='replicate')
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act2 = nn.SiLU()

        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        h = self.act1(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act2(h + self.shortcut(x))

class AttentionBlock(nn.Module):
    """
    A simple self-attention block for 3D feature maps.
    Helps the model focus on more informative regions, which is useful for sparse data.
    """
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv3d(channels, channels * 3, kernel_size=1)
        self.proj_out = nn.Conv3d(channels, channels, kernel_size=1)

    def forward(self, x):
        b, c, d, h, w = x.shape
        x_norm = self.norm(x)
        q, k, v = self.qkv(x_norm).chunk(3, dim=1)

        q = q.reshape(b, c, d * h * w)
        k = k.reshape(b, c, d * h * w)
        v = v.reshape(b, c, d * h * w)

        attn = torch.einsum('bci,bcj->bij', q, k) * (c ** -0.5)
        attn = F.softmax(attn, dim=-1)

        out = torch.einsum('bij,bcj->bci', attn, v)
        out = out.reshape(b, c, d, h, w)

        return x + self.proj_out(out)

class SNR_Embed(nn.Module):
    """
    Embeds the scalar SNR value into a feature vector.
    *FIX APPLIED: Configured for FiLM (outputs 2x dimensions for scale and shift)*
    """
    def __init__(self, out_dim):
        super().__init__()
        # Intermediate dimension can be anything, typically matches out_dim or similar
        hidden_dim = out_dim // 2 
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, snr_db):
        # snr_db is expected to be of shape (B, 1)
        return self.net(snr_db)

# =================================================================================
# 3D Autoencoder Architecture
# =================================================================================

class Encoder(nn.Module):
    def __init__(self, input_shape, base_channels=32, latent_channels=64, num_res_blocks=2, num_res_blocks_lidar=2):
        super().__init__()

        # --- Main 3D Branch ---
        self.init_conv = nn.Conv3d(1, base_channels, kernel_size=3, padding=1, padding_mode='replicate')

        self.down_blocks = nn.ModuleList()
        # stage 1
        in_ch = base_channels
        out_ch = base_channels * 2
        for _ in range(num_res_blocks_lidar):
            self.down_blocks.append(ResidualBlock(in_ch, out_ch))
            in_ch = out_ch  # after first block, in=out
        self.down_blocks.append(
            nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=(2, 2, 2), padding=1, padding_mode='replicate')
        )

        # stage 2
        in_ch = out_ch
        out_ch = base_channels * 4
        for _ in range(num_res_blocks_lidar):
            self.down_blocks.append(ResidualBlock(in_ch, out_ch))
            in_ch = out_ch
        self.down_blocks.append(
            nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=(2, 2, 1), padding=1, padding_mode='replicate')
        )
        
        # --- SNR Conditioning ---
        # *FIX APPLIED: Output is 2 * channels to support FiLM (gamma and beta)*
        self.snr_embed = SNR_Embed(out_dim=base_channels * 4 * 2)

        # --- Fusion and Final Blocks ---
        self.fusion_blocks = nn.ModuleList()
        current_channels = base_channels * 4
        for _ in range(num_res_blocks):
            self.fusion_blocks.append(ResidualBlock(current_channels, current_channels))
        self.fusion_blocks.append(AttentionBlock(current_channels))
        
        # Final convolution to shape the latent space
        self.final_conv = nn.Conv3d(current_channels, latent_channels, kernel_size=3, padding=1, padding_mode='replicate')

    def forward(self, x, snr_db):
        # Main branch
        h = self.init_conv(x)
        for block in self.down_blocks:
            h = block(h)
        
        # SNR branch - *FIX APPLIED: FiLM Conditioning*
        b, c, d, height, w = h.shape
        snr_embedding = self.snr_embed(snr_db).view(b, c * 2, 1, 1, 1)
        gamma, beta = snr_embedding.chunk(2, dim=1)
        
        # Apply scale (gamma) and shift (beta)
        h = h * (1 + gamma) + beta

        # Final processing
        for block in self.fusion_blocks:
            h = block(h)
        
        latent = self.final_conv(h)
        
        return latent

class Decoder(nn.Module):
    def __init__(self, output_shape, base_channels=32, latent_channels=64, num_res_blocks=2, num_res_blocks_lidar=2):
        super().__init__()

        current_channels = base_channels * 4
        self.init_conv = nn.Conv3d(latent_channels, current_channels, kernel_size=3, padding=1, padding_mode='replicate')
        
        # *FIX APPLIED: Output is 2 * channels to support FiLM*
        self.snr_embed = SNR_Embed(out_dim=current_channels * 2)

        self.res_blocks = nn.ModuleList()
        for _ in range(num_res_blocks):
            self.res_blocks.append(ResidualBlock(current_channels, current_channels))
        self.res_blocks.append(AttentionBlock(current_channels))

        # Upsampling blocks
        self.up_blocks = nn.ModuleList()

        # stage 1 - *FIX APPLIED: Upsample + Conv3d instead of ConvTranspose3d*
        out_ch = base_channels * 2
        self.up_blocks.append(
            nn.Sequential(
                nn.Upsample(scale_factor=(2, 2, 1), mode='trilinear', align_corners=False),
                nn.Conv3d(current_channels, out_ch, kernel_size=3, padding=1, padding_mode='replicate')
            )
        )
        for _ in range(num_res_blocks_lidar):
            self.up_blocks.append(ResidualBlock(out_ch, out_ch))

        # stage 2 - *FIX APPLIED: Upsample + Conv3d instead of ConvTranspose3d*
        in_ch = out_ch
        out_ch = base_channels
        self.up_blocks.append(
            nn.Sequential(
                nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=False),
                nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, padding_mode='replicate')
            )
        )
        for _ in range(num_res_blocks_lidar):
            self.up_blocks.append(ResidualBlock(out_ch, out_ch))

        # Final convolution to match output channels (1) and apply activation
        self.final_conv = nn.Conv3d(base_channels, 1, kernel_size=3, padding=1, padding_mode='replicate')

    def forward(self, x, snr_db):
        h = self.init_conv(x)
        b, c, d, height, w = h.shape
        
        # SNR branch - *FIX APPLIED: FiLM Conditioning*
        snr_embedding = self.snr_embed(snr_db).view(b, c * 2, 1, 1, 1)
        gamma, beta = snr_embedding.chunk(2, dim=1)
        h = h * (1 + gamma) + beta

        for block in self.res_blocks:
            h = block(h)

        for block in self.up_blocks:
            h = block(h)

        recon = F.softplus(self.final_conv(h))
        # recon = self.final_conv(h)
        return recon

class JSCC_Autoencoder(nn.Module):
    """
    The main autoencoder model. It normalizes the latent representation before the
    channel and simulates the AWGN channel.
    """
    def __init__(self, config):
        super().__init__()
        # Calculate latent shape to meet compression ratio
        total_input_elements = config["input_shape"][0] * config["input_shape"][1] * config["input_shape"][2]
        target_latent_elements = total_input_elements / config["compression_ratio"]
        latent_d = config["input_shape"][0] // 4
        latent_h = config["input_shape"][1] // 4
        latent_w = config["input_shape"][2] // 2
        
        latent_spatial_elements = latent_d * latent_h * latent_w
        latent_channels = int(target_latent_elements / latent_spatial_elements)
        
        print(f"Input elements: {total_input_elements}")
        print(f"Target latent elements for R={config['compression_ratio']}: {target_latent_elements}")
        print(f"Calculated latent shape: ({latent_channels}, {latent_d}, {latent_h}, {latent_w})")
        
        self.encoder = Encoder(
            input_shape=config["input_shape"],
            base_channels=config["base_channels"],
            latent_channels=latent_channels,
            num_res_blocks=config["num_res_blocks"],
            num_res_blocks_lidar=config["num_res_blocks_lidar"]
        )
        self.decoder = Decoder(
            output_shape=config["input_shape"],
            base_channels=config["base_channels"],
            latent_channels=latent_channels,
            num_res_blocks=config["num_res_blocks"],
            num_res_blocks_lidar=config["num_res_blocks_lidar"]
        )

    def forward(self, x, snr_db):
        # 1. Encode the input
        latent = self.encoder(x, snr_db)
        
        # 2. Normalize latent representation to have unit power
        b, c, d, h, w = latent.shape
        power = torch.mean(latent.pow(2), dim=[1,2,3,4], keepdim=True)
        latent_norm = latent / torch.sqrt(power + 1e-4)
        
        # 3. Pass through the AWGN channel
        snr_linear = 10.0 ** (snr_db / 10.0)
        noise_std = torch.sqrt(1.0 / (snr_linear))
        
        noise = torch.randn_like(latent_norm) * noise_std.view(b, 1, 1, 1, 1)
        latent_noisy = latent_norm + noise
        
        # 4. Decode the noisy latent representation
        reconstruction = self.decoder(latent_noisy, snr_db)

        return reconstruction

if __name__ == "__main__":

    CONFIG = {
        "input_shape": (128, 32, 16),
        "base_channels": 32,
        "num_res_blocks": 6,
        "num_res_blocks_lidar": 1,
        "channel_snr_db_range": (0, 20), # SNR range in dB for training
        "compression_ratio": 8,
        "learning_rate": 1e-4,
        "batch_size": 64,
        "num_epochs": 500,
        "results_dir": "./results",
        "validation_split": 0.2,
        "use_posterior_mean": True, # Whether to use posterior mean in the loss
    }

    model = JSCC_Autoencoder(CONFIG)

    test_input = torch.randn(2, 1, 128, 32, 16)
    test_snr = torch.tensor([10.0, 20.0]).unsqueeze(1)
    reconstructed = model(test_input, test_snr)
    print("Input shape:", test_input.shape)
    print("SNR shape:", test_snr.shape)
    print("Reconstructed shape:", reconstructed.shape)