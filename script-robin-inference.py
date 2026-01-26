import material
import robin

hub = material.Hub()
test = hub.getTest(batch=1, reproducibility=True)

device = 'cuda'
model = robin.Model(device)
model.activateLayer()

tag = 'robin-v1.0.0'
model.loadVersion(tag)
# model.loadWeight(path='log/robin-beta-0116/weight/10000.pt')

history = './robin-v1.0.0/' #
framework = robin.Framework(model, device, history)

for index, batch in enumerate(test):
    archive = f'{index}.jpg'
    comparison = False
    framework.makeInference(batch)
    framework.saveInference(archive, comparison)
    continue
