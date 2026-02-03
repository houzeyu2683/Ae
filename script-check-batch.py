import material

device = 'cuda'
batch = 2
hub = material.Hub(device)
data = hub.getData(batch)
index = 0
while(True):
    for b in data:
        continue
    print(f'finish {index} epoch')
    index += 1
    continue

