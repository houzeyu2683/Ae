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
        layer = diffusers.AutoencoderKL(
            in_channels=3,
            out_channels=3,
            latent_channels=8,
            block_out_channels=[32, 64, 128, 256, 256],  # 多一層 downsampling
            down_block_types=(
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D"
            ),
            up_block_types=(
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
                "UpDecoderBlock2D"
            ),
            layers_per_block=1
        )
        self.layer = layer.to(self.device)
        return(True)

    def getDistribution(
        self, 
        image: torch.Tensor
    ) -> tensordict.TensorDict:
        image = image.to(self.device, non_blocking=True)
        # assert hasattr(self.layer, 'encode')
        compression = self.layer.encode(image)
        mean = getattr(compression['latent_dist'], 'mean')
        variance = getattr(compression['latent_dist'], 'var')
        getSample = getattr(compression['latent_dist'], 'sample')
        sample = getSample()
        distribution = tensordict.TensorDict(device=self.device)
        distribution.set("mean", mean)
        distribution.set("variance", variance)
        distribution.set("sample", sample)
        return(distribution)
    
    def getReconstruction(
        self, 
        sample: torch.Tensor
    ) -> torch.Tensor:
        sample = sample.to(self.device, non_blocking=True)
        # assert hasattr(self.layer, 'decode')
        decompression = self.layer.decode(sample)
        reconstruction = getattr(decompression, 'sample')
        return(reconstruction)

    def getCriteria(
        self, 
        batch: tensordict.TensorDict,
        scale: float
    ) -> tensordict.TensorDict:
        batch = batch.to(self.device, non_blocking=True)
        image = batch['image']
        distribution = self.getDistribution(image)
        mean = distribution['mean']
        variance = distribution['variance']
        distance = -0.5 * torch.sum(
            1 + variance.log() - mean.pow(2) - variance,
            dim=(1, 2, 3)   # sum over (C, H, W)
        )
        divergence = distance.mean()
        sample = distribution['sample']
        reconstruction = self.getReconstruction(sample)
        pixel = torch.mean(
            (image-reconstruction).pow(2)
        )
        total = (scale * divergence) + pixel
        criteria = tensordict.TensorDict(device=self.device)
        criteria.set("divergence", divergence)
        criteria.set("pixel", pixel)
        criteria.set("total", total)
        return(criteria)

    # def getPerspective(self, number: int) -> torch.Tensor:
    #     shape = (number, 8, 4, 4)
    #     sample = torch.randn(*shape, device=self.device)
    #     generation = self.getReconstruction(sample)
    #     # route = getattr(self.layer, 'decode')
    #     # generation = route(noise)['sample']
    #     return(generation)

    # def getRepresentation(self, image: torch.Tensor) -> torch.Tensor:
    #     image = image.to(self.device, non_blocking=True)
    #     distribution = self.getDistribution(image)
    #     representation = distribution['mean'].flatten(1, -1)
    #     return(representation)

    forward = getCriteria
    pass



    # def getEstimation(
    #     self, 
    #     batch: tensordict.TensorDict
    # ) -> tensordict.TensorDict:
    #     estimation = tensordict.TensorDict(device=self.device)
    #     image = batch['image']
    #     distribution = self.getDistribution(image)
    #     reparameterization = distribution['reparameterization']
    #     reconstruction = self.getReconstruction(reparameterization)
    #     estimation.set('distribution', distribution)
    #     estimation.set('reconstruction', reconstruction)
    #     return(estimation)
        # estimation = self.getEstimation(batch)
        # # image = batch['image']
        # mean = estimation['distribution', 'mean']
        # variance = estimation['distribution','variance']
        # distance = -0.5 * torch.sum(
        #     1 + variance.log() - mean.pow(2) - variance,
        #     dim=(1, 2, 3)   # sum over (C, H, W)
        # )
        # divergence = scale * distance.mean()
        # pixel = torch.sum((image-reconstruction)**2, dim=[1,2,3]).mean()
        # image = batch['image']
        # reconstruction = estimation['reconstruction']
