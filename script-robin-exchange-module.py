import robin
import application
import material

device = 'cuda'
model = robin.Model(device)
model.activateLayer()

path = './log/robin-2026-0203/weight/130000.pt'
model.loadWeight(path)
model.eval()

hub = material.Hub()
number = 1
batch = hub.getBatch(number)
image = batch['image']
representation = model.getRepresentation(image)
#
luggage = application.Luggage(folder='./log/robin-2026-0203')
#
data = [image]
key = ['image']
luggage.exportModule(
    model=model, 
    method='getRepresentation', 
    data=data, 
    key=key, 
    archive='getRepresentation.onnx'
)
#
quantization = representation['quantization']
data = [quantization]
key = ['quantization']
luggage.exportModule(
    model=model, 
    method='getReconstruction', 
    data=data, 
    key=key, 
    archive='getReconstruction.onnx'
)