# from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import tensorboard.backend.event_processing.event_accumulator
import io
import PIL.Image

ea = tensorboard.backend.event_processing.event_accumulator.EventAccumulator("./log/robin-01030705/", size_guidance={"images": 1e8})
ea.Reload()
lock = ea.Images("Validation/Overview")
# loss = ea.Scalars("train/loss")

image = PIL.Image.open(io.BytesIO(lock[0].encoded_image_string))
image.save('test.png')


tensorboard.backend.event_processing.event_accumulator

import imageio
import numpy as np

def save_gif_with_pause(frames, path, fps=6, pause_sec=2.0):
    """
    frames: numpy array, shape (T, H, W, 3), uint8
    fps: 播放幀率
    pause_sec: 最後一幀停幾秒
    """
    pause_frames = int(fps * pause_sec)

    last = frames[-1:]
    frames_with_pause = np.concatenate(
        [frames, np.repeat(last, pause_frames, axis=0)],
        axis=0
    )

    imageio.mimsave(path, frames_with_pause, fps=fps)

# 使用
save_gif_with_pause(video_np, "result.gif", fps=6, pause_sec=2)

# import torch
# import torch.nn as nn
# from torch.optim import Adam
# from diffusers import VQModel

# # -----------------------
# # 超參數
# # -----------------------
# batch_size = 4
# channels = 3
# height = 64
# width = 64
# lr = 2e-4

# # -----------------------
# # 模型初始化
# # -----------------------
# vq_model = VQModel(
#     in_channels=3,
#     out_channels=3,
#     down_block_types=["DownEncoderBlock2D"] * 3,
#     up_block_types=["UpDecoderBlock2D"] * 3,
#     block_out_channels=(128, 256, 512),
#     layers_per_block=1,
#     # act_fn="silu",
#     # latent_channels=8,
#     # norm_num_groups=32,
#     num_vq_embeddings=512,  # codebook size
#     vq_embed_dim=8,
#     # scaling_factor=0.18215,
# )

# # 設定 optimizer
# optimizer = Adam(vq_model.parameters(), lr=lr)

# # 模擬訓練資料 (batch of images)
# x = torch.randn(batch_size, channels, height, width)

# # -----------------------
# # Training step
# # -----------------------
# vq_model.train()  # 訓練模式

# optimizer.zero_grad()

# # Forward pass (VQModel 會自動處理 encode -> quantize -> decode)
# output = vq_model(x)

# # output 包含:
# # - sample: 重建的圖片
# # - commit_loss: VQ 的 commitment loss
# x_recon = output.sample

# # Reconstruction loss
# recon_loss = nn.MSELoss()(x_recon, x)

# # VQ commitment loss (從 output 取得)
# commit_loss = output.commit_loss

# ##
# ## 有辦法透過output計算commit_loss？
# ##

# # 總 loss
# total_loss = recon_loss + commit_loss

# # Backward
# total_loss.backward()

# # 更新參數
# optimizer.step()

# print("Reconstruction loss:", recon_loss.item())
# print("Commit loss:", commit_loss.item())
# print("Total loss:", total_loss.item())
