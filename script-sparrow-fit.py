import material
import sparrow

hub = material.Hub()
data = hub.getData(batch=256)
validation = hub.getValidation(batch=32, reproducibility=False)

device = 'cuda'
model = sparrow.Model(device)
model.activateLayer()
# model.loadWeight(path='./exp/Dec23-2/weight/400.pt') # 看情況

history = './log/sparrow-2026-0117'
framework = sparrow.Framework(model, device, history)

snapshot = 10000
total = -1
accumulation = 1
framework.fitWeight(data, snapshot, total, accumulation, validation)
