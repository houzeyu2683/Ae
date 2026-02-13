import torch
import diffusers
import tensordict
import safetensors.torch
import os
import requests

class Model(torch.nn.Module):

    def __init__(self, device: str) -> None:
        super().__init__()
        self.device = device
        return

    def loadWeight(self, path: str) -> bool:
        weight = safetensors.torch.load_file(path)
        self.load_state_dict(weight)
        return(True)

    def activateLayer(self) -> bool:
        layer = diffusers.AutoencoderKL(
            in_channels=3,
            out_channels=3,
            latent_channels=32,
            block_out_channels=[32, 64, 128, 256],  # 多一層 downsampling
            down_block_types=(
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                # "DownEncoderBlock2D"
            ),
            up_block_types=(
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
                # "UpDecoderBlock2D"
            ),
            layers_per_block=1
        )
        self.layer = layer.to(self.device)
        return(True)

    def getCompression(
        self, 
        image: torch.Tensor
    ) -> torch.Tensor:
        image = image.to(self.device, non_blocking=True)
        # assert hasattr(self.layer, 'encode')
        projection = self.layer.encode(image)
        compression = getattr(projection['latent_dist'], 'mean')
        return(compression)

    def getReconstruction(
        self, 
        compression: torch.Tensor
    ) -> torch.Tensor:
        compression = compression.to(self.device, non_blocking=True)
        # assert hasattr(self.layer, 'decode')
        decompression = self.layer.decode(compression)
        reconstruction = getattr(decompression, 'sample')
        return(reconstruction)

    def getCriteria(
        self, 
        batch: tensordict.TensorDict,
        # scale: float
    ) -> tensordict.TensorDict:
        batch = batch.to(self.device, non_blocking=True)
        image = batch['image']
        # distribution = self.getDistribution(image)
        compression = self.getCompression(image)
        reconstruction = self.getReconstruction(compression)
        pixel = torch.mean(
            (image-reconstruction).pow(2)
        )
        # total = (scale * divergence) + pixel
        total = pixel
        criteria = tensordict.TensorDict(device=self.device)
        # criteria.set("divergence", divergence)
        criteria.set("pixel", pixel)
        criteria.set("total", total)
        return(criteria)

    forward = getCriteria
    pass
