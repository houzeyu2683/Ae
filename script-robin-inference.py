import material
import robin

hub = material.Hub()
test = hub.getTest(batch=1, reproducibility=True)

device = 'cuda'
model = robin.Model(device)
model.activateLayer()
model.loadWeight(path='log/robin-beta-0116/weight/10000.pt')

history = './log/robin-beta-0116/' #
framework = robin.Framework(model, device, history)

for index, batch in enumerate(test):
    archive = f'{index}.jpg'
    comparison = False
    framework.makeInference(batch)
    framework.saveInference(archive, comparison)
    continue
