import material
import robin

hub = material.Hub()
data = hub.getData(number=128)
validation = hub.getValidation(number=32, reproducibility=False)

device = 'cuda'
model = robin.Model(device)
model.activateLayer()
# model.loadWeight(path='./log/robin-2026-0103/weight/60900.pt') # 看情況

history = './log/robin-2026-0203' #
framework = robin.Framework(model, device, history)

snapshot = 10000
total = -1
accumulation = 4
framework.fitWeight(data, snapshot, total, accumulation, validation)
