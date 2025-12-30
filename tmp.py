import torch
import torch.nn as nn
from torch.optim import Adam
from diffusers import VQModel

# -----------------------
# 超參數
# -----------------------
batch_size = 4
channels = 3
height = 64
width = 64
lr = 2e-4

# -----------------------
# 模型初始化
# -----------------------
vq_model = VQModel(
    block_out_channels=[128, 256],
    in_channels=3,
    out_channels=3,
    downsample_times=2,
    latent_channels=64,
    num_vq_embeddings=512,
    vq_embedding_dim=64,
)

# 設定 optimizer
optimizer = Adam(vq_model.parameters(), lr=lr)

# 模擬訓練資料 (batch of images)
x = torch.randn(batch_size, channels, height, width)

# -----------------------
# Training step
# -----------------------
vq_model.train()  # 訓練模式

optimizer.zero_grad()

# 1. Encoder
z_e = vq_model.encoder(x)  # [B, C_latent, H_latent, W_latent]

# 2. Vector Quantization
z_q, vq_loss = vq_model.vq(z_e)  # 返回量化 latent 和 VQ loss

# 3. Decoder
x_recon = vq_model.decoder(z_q)

# 4. Reconstruction loss
recon_loss = nn.MSELoss()(x_recon, x)

# 5. 總 loss
total_loss = recon_loss + vq_loss

# 6. Backward
total_loss.backward()

# 7. 更新參數
optimizer.step()

print("Reconstruction loss:", recon_loss.item())
print("VQ loss:", vq_loss.item())
print("Total loss:", total_loss.item())
