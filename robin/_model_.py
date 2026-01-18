import torch
import diffusers
import tensordict
import safetensors.torch

class Model(torch.nn.Module):

    def __init__(self, device: str) -> None:
        super().__init__()
        self.device = device
        return

    def loadWeight(self, path: str) -> bool:
        state_dict = safetensors.torch.load_file(path)
        self.load_state_dict(state_dict)
        return(True)
    
    def activateLayer(self) -> bool:
        layer = diffusers.VQModel(
            in_channels=3,
            out_channels=3,
            down_block_types=(
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                # "DownEncoderBlock2D"
            ),
            up_block_types=(
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
                # "UpDecoderBlock2D"
            ),
            block_out_channels=(32, 64),
            layers_per_block=1,
            num_vq_embeddings=64,  # codebook size
            vq_embed_dim=8
        )
        self.layer = layer.to(self.device)
        return(True)

    def getRepresentation(
        self, 
        image: torch.Tensor
    ) -> tensordict.TensorDict:
        image = image.to(self.device, non_blocking=True)
        embedding = self.layer.encode(image).latents
        quantization, _, (_, _, token) = self.layer.quantize(embedding) # quantization
        representation = tensordict.TensorDict(device=self.device)
        representation.set('embedding', embedding)
        representation.set('quantization', quantization)
        representation.set('token', token)
        return(representation)

    def getReconstruction(self, quantization: torch.Tensor) -> torch.Tensor:
        quantization = quantization.to(self.device, non_blocking=True)
        value = self.layer.post_quant_conv(quantization)
        reconstruction = self.layer.decoder(value)
        return(reconstruction)

    def getCriteria(
        self, 
        batch: tensordict.TensorDict,
    ) -> tensordict.TensorDict:
        batch = batch.to(self.device, non_blocking=True)
        image = batch['image']
        node = self.layer(image)
        commitment = getattr(node, 'commit_loss')
        reconstruction = getattr(node, 'sample')
        torch.nn.functional.tanh(reconstruction)
        pixel = torch.abs(image - reconstruction).mean()
        pixel = torch.mean(
            torch.pow(image-reconstruction, 2)
        ).sum()
        # brightness = torch.sub(
        #     self.getLuminosity(image),
        #     self.getLuminosity(reconstruction)
        # )
        # brightness = torch.pow(brightness, 2).mean()
        total = commitment + pixel #+ brightness
        criteria = tensordict.TensorDict(device=self.device)
        criteria.set('commitment', commitment)
        criteria.set('pixel', pixel)
        # criteria.set('brightness', brightness)
        criteria.set('total', total)
        return(criteria)

    # def getLuminosity(self, image: torch.Tensor) -> torch.Tensor:
    #     """
    #     img: torch.Tensor, shape (N, 3, H, W), value range (-1, 1)
    #     return: torch.Tensor, shape (N, 3, H, W)
    #             Y in [0,1], U/V roughly in [-0.5,0.5]
    #     """
    #     image = image.to(self.device, non_blocking=True)
    #     # (-1,1) → (0,1)
    #     color = (image + 1.0) / 2.0

    #     red = color[:, 0:1]
    #     green = color[:, 1:2]
    #     blue = color[:, 2:3]

    #     # Standard YUV (BT.601-like)
    #     luminosity = 0.299 * red + 0.587 * green + 0.114 * blue
    #     # u = -0.14713 * r - 0.28886 * g + 0.436 * b
    #     # v = 0.615 * r - 0.51499 * g - 0.10001 * b

    #     # yuv = torch.cat([y, u, v], dim=1)
    #     return(luminosity)

    forward = getCriteria
    pass


