推理：
x_now (continuous latent)
    ↓
預測模型
    ↓
x_{now+t} (continuous latent)
    ↓
VQVAE decoder
    ↓
圖片



方案 1（推薦）：

✅ 小 β (0.001-0.01)
✅ temporal consistency loss
✅ conditional VAE (從 N(x_now, σ) 採樣)

# Loss
recon = ||x_pred - x_gt||²
kl = KL(N(μ,σ) || N(x_now, σ_base))  # 以當前幀為先驗
temporal = ||x_{t+1} - x_t||²

loss = recon + 0.01*kl + λ*temporal
方案 2（更保守）：

✅ 漸進式 σ (t 小 → σ 小)
✅ temporal consistency
✅ 小 β
這樣：

短期預測穩定流暢
長期有多樣性
不會暈開
λ 建議 0.1-1.0，實驗調整。

要這樣試試嗎？