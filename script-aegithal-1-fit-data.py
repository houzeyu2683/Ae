import material
import aegithal

hub = material.Hub()
data = hub.getData(number=256)
validation = hub.getValidation(number=32, reproducibility=False)

device = 'cuda'
model = aegithal.Model(device)
model.activateLayer()
model.loadCheckpoint(path='./log/aegithal-2026-0117/checkpoint/210000.pt') # 看情況

history = './log/aegithal-2026-0208'
framework = aegithal.Framework(model, device, history)

snapshot = 1000
total = -1
accumulation = 4
framework.fitWeight(data, snapshot, total, accumulation, validation)
