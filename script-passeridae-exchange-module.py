import passeridae
import application
import material

device = 'cuda'
model = passeridae.Model(device)
model.activateLayer()

path = './log/passeridae-2026-0117/weight/210000.pt'
model.loadWeight(path)
model.eval()

hub = material.Hub()
number = 1
batch = hub.getBatch(number)
image = batch['image']
compression = model.getCompression(image)
#
luggage = application.Luggage(folder='./log/passeridae-2026-0117/')
#
data = [image]
key = ['image']
luggage.exportModule(
    model=model, 
    method='getCompression', 
    data=data, 
    key=key, 
    archive='getCompression.onnx'
)
#
data = [compression]
key = ['compression']
luggage.exportModule(
    model=model, 
    method='getReconstruction', 
    data=data, 
    key=key, 
    archive='getReconstruction.onnx'
)