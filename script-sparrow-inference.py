import material
import sparrow

hub = material.Hub()
test = hub.getTest(batch=32, reproducibility=True)

device = 'cuda'
model = sparrow.Model(device)
model.activateLayer()
model.loadWeight(path='log/sparrow-2026-0116/weight/80000.pt')

history = './log/sparrow-2026-0116/' #
framework = sparrow.Framework(model, device, history)

batch = next(iter(test))
framework.makeComparison(batch)
framework.saveComparison(archive='80000.jpg')
number = 64
framework.makePerspective(number)
framework.savePerspective(archive='80000.jpg')


