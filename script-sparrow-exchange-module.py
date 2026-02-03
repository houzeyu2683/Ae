import sparrow
import application
import material

device = 'cuda'
model = sparrow.Model(device)
model.activateLayer()

path = './log/sparrow-2026-0117/weight/210000.pt'
model.loadWeight(path)

# tag = 'sparrow-v1.0.0'
# model.loadVersion(tag)

hub = material.Hub()
number = 1
batch = hub.getBatch(number)
image = batch['image']
compression = model.getCompression(image)
#
luggage = application.Luggage(folder='./log/sparrow-2026-0117/')
#
data = [image]
key = ['image']
# elasticity = {'image': [0]}
luggage.exportModule(
    model=model, 
    method='getCompression', 
    data=data, 
    key=key, 
    archive='getCompression.onnx',
    # elasticity=elasticity
)
#
data = [compression]
key = ['compression']
# elasticity = {'compression': [0]}
luggage.exportModule(
    model=model, 
    method='getReconstruction', 
    data=data, 
    key=key, 
    archive='getReconstruction.onnx',
    # elasticity=elasticity
)