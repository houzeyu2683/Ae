import material
import facility

device = 'cuda'
hub = material.Hub(device)
batch = 256
data = hub.getData(batch)

validation = hub.getValidation(
    batch=32,
    reproducibility=False
)

sparrow = facility.Sparrow(device)
sparrow.activateLayer()

history = './exp/Dec31'
framework = facility.Framework(sparrow, device, history)
# framework.loadWeight(path='./exp/Dec23-2/weight/400.pt') # 看情況

snapshot = 1000
total = -1
accumulation = 1
framework.fitWeight(data, snapshot, total, accumulation, validation)
