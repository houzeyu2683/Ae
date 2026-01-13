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
# model.loadWeight(path='./log/robin-2026-0103/weight/60900.pt') # 看情況

history = './log/robin-unit-test' #
framework = robin.Framework(model, device, history)

snapshot = 2000
total = -1
accumulation = 4
framework.fitWeight(data, snapshot, total, accumulation, validation)
