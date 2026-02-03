import torch
import sparrow

device = 'cuda'
tag = 'sparrow-v1.0.0'
model = sparrow.Model(device)
model.activateLayer()
model.loadVersion(tag)
model.eval()

shape = (1, 3, 64, 64)
data = tuple([torch.randn(shape)])

network = torch.nn.Module()
# network.forward = model.getCompression
setattr(network, 'forward', model.getCompression)
# network(image)

# 用 lambda 包起來
torch.onnx.export(
    network,
    data,
    "model.onnx",
    input_names=["image"],
)