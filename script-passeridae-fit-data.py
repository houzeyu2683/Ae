import material
import passeridae

hub = material.Hub()
data = hub.getData(number=256)
validation = hub.getValidation(number=32, reproducibility=False)

device = 'cuda'
model = passeridae.Model(device)
model.activateLayer()
# model.loadWeight(path='./exp/Dec23-2/weight/400.pt') # 看情況

history = './log/passeridae-2026-0117'
framework = passeridae.Framework(model, device, history)

snapshot = 10000
total = -1
accumulation = 1
framework.fitWeight(data, snapshot, total, accumulation, validation)
