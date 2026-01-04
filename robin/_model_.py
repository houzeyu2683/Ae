import torch
import diffusers
import tensordict

class Model(torch.nn.Module):

    def __init__(self, device: str) -> None:
        super().__init__()
        self.device = device
        return

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
        representation = tensordict.TensorDict(device=self.device)
        space = self.layer.encode(image).latents
        embedding, _, (_, _, token) = self.layer.quantize(space)
        representation.set('embedding', embedding)
        representation.set('token', token)
        return(representation)

    def getReconstruction(self, embedding: torch.Tensor) -> torch.Tensor:
        embedding = self.layer.post_quant_conv(embedding)
        reconstruction = self.layer.decoder(embedding)
        return(reconstruction)

    def getCriteria(
        self, 
        batch: tensordict.TensorDict,
    ) -> tensordict.TensorDict:
        criteria = tensordict.TensorDict(device=self.device)
        image = batch['image']
        node = self.layer(image)
        commitment = getattr(node, 'commit_loss')
        reconstruction = getattr(node, 'sample')
        pixel = torch.abs(image-reconstruction).mean()
        pixel = torch.mean(
            torch.pow(image-reconstruction, 2)
        ).sum()
        total = commitment + pixel
        criteria.set('commitment', commitment)
        criteria.set('pixel', pixel)
        criteria.set('total', total)
        # criteria.set('reconstruction', reconstruction) #!
        return(criteria)

    forward = getCriteria
    pass


