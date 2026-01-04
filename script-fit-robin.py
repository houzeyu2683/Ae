import material
import robin

device = 'cuda'
hub = material.Hub(device)
batch = 256
data = hub.getData(batch)

validation = hub.getValidation(
    batch=32,
    reproducibility=False
)

model = robin.Model(device)
model.activateLayer()

history = './log/robin-01040703'
framework = robin.Framework(model, device, history)
# framework.loadWeight(path='./exp/Dec23-2/weight/400.pt') # 看情況

snapshot = 2000
total = -1
accumulation = 4
framework.fitWeight(data, snapshot, total, accumulation, validation)
