import aegithal

device = 'cpu'
model = aegithal.Model(device)
model.activateLayer()

history = './log/aegithal-2026-0117/'
framework = aegithal.Framework(model, device, history)

# # aggregate = framework.Aggregate(history)
checkpoint = [
    '210000.pt',
    '211000.pt',
    '212000.pt',
    '213000.pt',
]
framework.saveWeight(checkpoint)
