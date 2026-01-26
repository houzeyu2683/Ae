import material
import sparrow

hub = material.Hub()
test = hub.getTest(batch=1, reproducibility=True)

device = 'cuda'
model = sparrow.Model(device)
model.activateLayer()

tag = 'sparrow-v1.0.0'
model.loadVersion(tag)
# model.loadWeight(path='log/sparrow-2026-0117/weight/150000.pt')

history = './log/sparrow-from-hub/' #
framework = sparrow.Framework(model, device, history)

for index, batch in enumerate(test):
    framework.makeComparison(batch)
    framework.saveComparison(archive=f'{index}-sample.jpg')
    continue
# number = 64
# framework.makePerspective(number)
# framework.savePerspective(archive='80000.jpg')


