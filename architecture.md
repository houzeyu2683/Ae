# Latent Video Generation - Training Architecture

## Overview

在 latent space 中訓練自回歸模型，從起始幀生成自然影片。

```
z_0 → G 自回歸 → [z_0, z_1, ..., z_T] → Decoder(ONNX) → 影片
```

## Data

| 項目 | 規格 |
|------|------|
| 影片數量 | 200,000 |
| 每部影片幀數 | 29 |
| Latent shape | (32, 8, 8) |
| 每部影片 tensor | (29, 32, 8, 8) |
| 訓練 pair 數量 | 200,000 × 28 = 5,600,000 |
| 目標生成長度 | 5 秒 = 145 幀 |

## Model Components

### Encoder

將真實位移壓縮為低維 condition 向量。

```
輸入: Δz_t = z_{t+1} - z_t    shape: (32, 8, 8)
輸出: mu, logvar               shape: (dim_c,)
c = mu + exp(0.5 * logvar) * noise   (reparameterization trick)
```

### Generator (G)

以過去 k 幀為 context，搭配 condition c，預測下一步位移。

```
輸入: [z_{t-k}, ..., z_t]     shape: (k, 32, 8, 8) → flatten → (k, 2048)
      c                       shape: (dim_c,)
模型: Causal Transformer
輸出: Δz_t                    shape: (32, 8, 8)

z_{t+1} = z_t + Δz_t
```

Transformer 規格建議:
- 投影維度: 512
- 層數: 4-6
- Attention heads: 8
- Context window k: 16

### Discriminator (D)

判斷相鄰幀過渡是否自然。

```
輸入: (z_t, z_{t+1})          shape: (64, 8, 8)  (concat on channel dim)
模型: CNN
輸出: score (real / fake)
```

## Training

### 階段一: CVAE (Encoder + G)

先用 MSE + KL 學會基本動態預測。

```
真實 Δz_t → Encoder → c (mu, logvar)
G([z_{t-k}, ..., z_t], c) → Δz_pred

loss = MSE(Δz_pred, Δz_t) + β * KL(mu, logvar || N(0, I))
```

- MSE: 重建真實位移
- KL: 約束 c 的分佈接近 N(0, I)，確保推論時可以 sample

### 階段二: CVAE-GAN (加入 D)

在階段一基礎上加入 D，提升生成品質。

**訓練 D:**
```
positive: D(z_t, z_{t+1}_real)  → 1
negative: D(z_t, z_{t+1}_fake)  → 0
更新 D 參數
```

**訓練 G + Encoder:**
```
D 凍結 (requires_grad = False，保留計算圖)
delta = G(z_t, c)
z_fake = z_t + delta
loss_G = MSE(delta, Δz_t) + β * KL + GAN_loss(-D(z_t, z_fake))
梯度穿過 D 回傳到 G 和 Encoder
更新 G 和 Encoder 參數
```

## Inference

```
1. 給定起始幀 z_0
2. sample c ~ N(0, I)
3. 自回歸生成:
   for t in range(T):
       Δz_t = G([z_{t-k}, ..., z_t], c)
       z_{t+1} = z_t + Δz_t
4. latent sequence → Decoder(ONNX) → 影片
```

- 調整 c 可控制生成方向
- c 從真實影片 encode 可複製動態風格
- c 隨機 sample 可生成多樣化影片

## Hardware

| 項目 | 估算 |
|------|------|
| 每幀資料量 | 32×8×8 = 2048 floats = 8KB |
| 模型參數量 | ~10-30M |
| 顯存需求 | < 1GB |
| 可用顯存 | 12GB |
| Batch size | 可開大 (64+) |
