import passeridae

device = 'cpu'
model = passeridae.Model(device)
model.activateLayer()

history = './log/passeridae-2026-0117/'
framework = passeridae.Framework(model, device, history)

# # aggregate = framework.Aggregate(history)
checkpoint = [
    '210000.pt',
    '211000.pt',
    '212000.pt',
    '213000.pt',
]
framework.saveWeight(checkpoint)
