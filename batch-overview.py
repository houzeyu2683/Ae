import material

device = 'cuda'
batch = 512
hub = material.Hub(device)
data = hub.getData(batch)
index = 0
while(True):
    for b in data:
        continue
    print(f'finish {index} epoch')
    index += 1
    continue
# print(next(iter(data)))
# validation = hub.getValidation(batch)
# print(next(iter(validation)))
# test = hub.getTest(batch)
# print(next(iter(test)))
# import itertools

# while(True):
# # iteration = enumerate(itertools.cycle(data), 0)
# for number, image in iteration:
#     print(number)
#     continue
