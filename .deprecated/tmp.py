import PIL.Image
import os
import glob
import torch
import numpy
import torchvision.transforms.functional
import plotly.express
import torch.functional

class Orientation:

    def __init__(self, image: torch.Tensor) -> None:
        self.image = image
        return
    
    def getGradient(self) -> torch.Tensor:
        # gradient = {
        #     'horizontal': frame[:, 1:-1, 2:] - frame[:, 1:-1, :-2],
        #     'vertical': frame[:, 2:, 1:-1] - frame[:, :-2, 1:-1],
        # }
        gradient = [
            self.image[:, 1:-1, 2:] - self.image[:, 1:-1, :-2],
            self.image[:, 2:, 1:-1] - self.image[:, :-2, 1:-1]
        ]
        gradient = torch.cat(gradient, dim=0)
        return(gradient)

    pass

loop  = glob.glob('./material/storage/T01/0/*.jpg')
# loop = [
#     './material/storage/T01/0/0.jpg',
#     './material/storage/T01/0/1.jpg',
#     './material/storage/T01/0/2.jpg',
#     './material/storage/T01/0/3.jpg'
# ]
x = []
y = []
z = []
c = []
for step, path in enumerate(loop):
    image = PIL.Image.open(path).convert("L").resize((64, 64))
    image = torchvision.transforms.functional.to_tensor(image)
    image = torchvision.transforms.functional.pad(image, [1,1,1,1])
    # frame = frame
    orientation = Orientation(image)
    gradient = orientation.getGradient()
    gradient = torch.nn.functional.avg_pool2d(gradient, (16, 16), (16, 16))
    timestep = torch.zeros((4, 4)) + step
    x += gradient[0,:,:].flatten().numpy().tolist()
    y += gradient[1,:,:].flatten().numpy().tolist()
    z += timestep.flatten().numpy().tolist()
    c += [str(index) for index in range(4*4)]
    continue
    # # mag = torch.sqrt(gradient[0,:,:]**2 + gradient[1,:,:]**2)
    # # ori = torch.atan2(gradient[0,:,:], gradient[1,:,:])
    # # le = torch.cos(ori)
    # if(index==0):
    #     pivot = gradient
    #     continue
    # # delta = gradient - pivot
    # gradient
    # # mag = torch.sqrt(Gx**2 + Gy**2)
# fig = plotly.express.line(x=z, y=y, color=c)
fig = plotly.express.scatter_3d(x=x, y=y, z=z, color=c)
fig.update_layout(
    scene=dict(
        # aspectmode='data'
        aspectmode='manual',
        aspectratio=dict(x=1, y=1, z=10)  # z 拉成 2 倍高
    )
)
fig.show()

    # torchvision.utils.save_image(
    #     # le,
    #     'le.jpg',
    #     value_range=(-1, 1), 
    #     normalize=True
    # )
    # torchvision.utils.save_image(
    #     gradient['vertical'],
    #     'vertical.jpg',
    #     value_range=(-1, 1), 
    #     normalize=True
    # )

    # pivot = gradient
    # continue

# import torch
# import safetensors.torch
# root = 'https://github.com/'

# # weight = torch.hub.load_state_dict_from_url(
# #     
# # )

# # print(weight)

# import requests
# import torch
# # from safetensors.torch import load_file
# import os
# import safetensors.torch

# root = './.hub/weight/'
# url='https://github.com/houzeyu2683/VAe/releases/download/robin-v1.0.0/weight.pt'
# folder = os.path.basename(os.path.dirname(url))
# archive = os.path.basename(url)
# path = os.path.join(root, folder, archive)
# os.makedirs(os.path.dirname(path), exist_ok=True)

# response = requests.get(url, stream=True)
# with open(path, 'wb') as f:
#     for chunk in response.iter_content(chunk_size=8192):
#         f.write(chunk)

# weight = safetensors.torch.load_file(path)



# # print('check')

# # def load_safetensors_from_url(url, local_path, device="cpu"):
# #     """Downloads a safetensors file from a URL and loads it."""

# #     # 1. Download the file
# #     print(f"Downloading from {url}...")
# #     try:
# #         response = requests.get(url, stream=True)
# #         response.raise_for_status() # Raise an exception for bad status codes

# #         with open(local_path, 'wb') as f:
# #             for chunk in response.iter_content(chunk_size=8192):
# #                 f.write(chunk)
# #         print(f"Downloaded and saved to {local_path}")

# #     except requests.exceptions.RequestException as e:
# #         print(f"Error during download: {e}")
# #         return None

# #     # 2. Load the safetensors file from the local path
# #     try:
# #         tensors = load_file(local_path, device=device)
# #         print(f"Successfully loaded tensors to {device} device.")
# #         return tensors
# #     except Exception as e:
# #         print(f"Error loading safetensors file: {e}")
# #         return None
# #     finally:
# #         # Optional: Remove the local file after loading
# #         # os.remove(local_path)
# #         pass

# # # Example Usage (replace with your actual URL and path)
# # # Note: A real model file URL from Hugging Face might look different.
# # # This is a generic example.
# # model_url = "https://huggingface.co" 
# # model_path = "downloaded_model.safetensors"

# # # Load to CPU or GPU (e.g., "cuda:0")
# # loaded_tensors = load_safetensors_from_url(model_url, model_path, device="cpu")

# # if loaded_tensors:
# #     print(f"Keys in the loaded state dict: {loaded_tensors.keys()}")