import material
import robin

hub = material.Hub()
test = hub.getTest(batch=1, reproducibility=True)

device = 'cuda'
model = robin.Model(device)
model.activateLayer()
model.loadWeight(path='./log/robin-2026-0103/weight/60000.pt')

history = './log/robin-2026-0103/' #
framework = robin.Framework(model, device, history)

for index, batch in enumerate(test):
    archive = f'{index}.jpg'
    comparison = False
    framework.makeInference(batch)
    framework.saveInference(archive, comparison)
    continue
