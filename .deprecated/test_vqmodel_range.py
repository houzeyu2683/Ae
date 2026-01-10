import torch
import diffusers

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 創建模型
model = diffusers.VQModel(
    in_channels=3,
    out_channels=3,
    down_block_types=("DownEncoderBlock2D",),
    up_block_types=("UpDecoderBlock2D",),
    block_out_channels=(32,),
    layers_per_block=1,
    num_vq_embeddings=16,
    vq_embed_dim=8
).to(device)

# 測試 [0, 1] 範圍
img_01 = torch.rand(1, 3, 64, 64).to(device)
output_01 = model(img_01).sample

# 測試 [-1, 1] 範圍
img_11 = torch.rand(1, 3, 64, 64).to(device) * 2 - 1
output_11 = model(img_11).sample

print(f"輸入 [0,1] 範圍:")
print(f"  輸入: min={img_01.min():.3f}, max={img_01.max():.3f}")
print(f"  輸出: min={output_01.min():.3f}, max={output_01.max():.3f}")

print(f"\n輸入 [-1,1] 範圍:")
print(f"  輸入: min={img_11.min():.3f}, max={img_11.max():.3f}")
print(f"  輸出: min={output_11.min():.3f}, max={output_11.max():.3f}")

# 檢查 decoder 最後一層
print(f"\nDecoder 最後一層:")
print(model.decoder.conv_out)
