import material
import application

import numpy
import torch
import torchvision.utils

interface = {}

version, archive = 'sparrow-v1.0.0', 'getCompression.onnx'
getCompression = application.Service(version, archive)
getCompression.loadSession()

version, archive = 'sparrow-v1.0.0', 'getReconstruction.onnx'
getReconstruction = application.Service(version, archive)
getReconstruction.loadSession()

hub = material.Hub()
batch = hub.getBatch(number=256)
#
group = []
for image in batch['image']:
    image = image[None, :, :, :].numpy()
    #
    request = {'image': image}
    response = getCompression(request)
    compression = response[0]
    #
    request = {"compression": compression}
    response = getReconstruction(request)
    reconstruction = response[0]
    #
    group += [image, reconstruction]
    continue
#
together = numpy.concatenate(group, axis=0)
torchvision.utils.save_image(
    torch.from_numpy(together),
    'check.jpg',
    value_range=(-1, 1), 
    normalize=True
)
# model = sparrow.Model(device)
# model.activateLayer()

# tag = 'sparrow-v1.0.0'
# model.loadVersion(tag)
# # model.loadWeight(path='log/sparrow-2026-0117/weight/150000.pt')

# history = './log/sparrow-from-hub/' #
# framework = sparrow.Framework(model, device, history)

# for index, batch in enumerate(test):
#     framework.makeComparison(batch)
#     framework.saveComparison(archive=f'{index}-sample.jpg')
#     continue
# # number = 64
# # framework.makePerspective(number)
# # framework.savePerspective(archive='80000.jpg')


