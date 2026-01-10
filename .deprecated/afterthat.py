import torch
import tensordict

feature = tensordict.TensorDict(
    {
        'embedding': torch.randn(12, 32, 16, 16),
        'quantization': torch.randn(12, 32, 16, 16),
        'amount': torch.randint(1, 1000+1, (12, 1)), # [low, high)
    }
)
target = tensordict.TensorDict(
    {
        'embedding': torch.randn(12, 32, 16, 16),
        'quantization': torch.randn(12, 32, 16, 16),
    }
)
batch = tensordict.TensorDict(
    {
        'feature': feature,
        'target': target
    },
    batch_size=12
)



vein = torch.nn
e = batch['feature', 'embedding'] # *, 32, 16, 16
e = vein.Conv2d(32, 32, kernel_size=3, padding=1)(e) # *, 32, 16, 16
e = vein.Conv2d(32, 32, kernel_size=3, padding=1)(e) # *, 32, 16, 16
a = batch['feature', 'amount']
t = vein.Embedding(1000+1, 32)(a) # *, 1, 32
t = torch.squeeze(t) # *, 32
t.shape
z = e + t[:, :, None, None]




