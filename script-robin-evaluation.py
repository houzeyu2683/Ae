import material
import robin

device = 'cuda'
hub = material.Hub(device)
test = hub.getTest(batch=1, reproducibility=True)

model = robin.Model(device)
model.activateLayer()
path = './log/robin-2026-0103/weight/60900.pt'
model.loadWeight(path)

history = './log/robin-2026-0103' #
framework = robin.Framework(model, device, history)

for index, batch in enumerate(test):
    framework.makeInference(batch)
    name = str(index)
    framework.saveInference(name)
    continue
