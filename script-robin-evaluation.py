import material
import robin

device = 'cuda'
hub = material.Hub(device)
batch = 1
test = hub.getTest(batch, reproducibility=True)

model = robin.Model(device)
model.activateLayer()

history = './log/eval' #
framework = robin.Framework(model, device, history)
path = './log/robin-2026-0103-1/weight/60900.pt'
framework.loadWeight(path) # 看情況

# framework.fitWeight(data, snapshot, total, accumulation, validation)
framework.makeEvaluation(test)