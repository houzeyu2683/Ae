import torch
import diffusers
import tensordict

class Model(torch.nn.Module):

    def __init__(self, device: str) -> None:
        super().__init__()
        self.device = device
        return

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
        self, image: torch.Tensor
    ) -> tensordict.TensorDict:
        distribution = tensordict.TensorDict(device=self.device)
        route = getattr(self.layer, 'encode')
        space = route(image)
        mean = getattr(space['latent_dist'], 'mean')
        variance = getattr(space['latent_dist'], 'var')
        getSpecimen = getattr(space['latent_dist'], 'sample')
        specimen = getSpecimen()
        distribution.set("mean", mean)
        distribution.set("variance", variance)
        distribution.set("specimen", specimen)
        return(distribution)
    
    def getReconstruction(
        self, 
        specimen: torch.Tensor
    ) -> torch.Tensor:
        route = getattr(self.layer, 'decode')
        node = route(specimen)
        reconstruction = getattr(node, 'sample')
        return(reconstruction)

    def getCriteria(
        self, 
        batch: tensordict.TensorDict,
        scale: float
    ) -> tensordict.TensorDict:
        criteria = tensordict.TensorDict(device=self.device)
        image = batch['image']
        distribution = self.getDistribution(image)
        mean = distribution['mean']
        variance = distribution['variance']
        distance = -0.5 * torch.sum(
            1 + variance.log() - mean.pow(2) - variance,
            dim=(1, 2, 3)   # sum over (C, H, W)
        )
        divergence = scale * distance.mean()
        specimen = distribution['specimen']
        reconstruction = self.getReconstruction(specimen)
        pixel = torch.mean(
            (image-reconstruction).pow(2)
        )
        total = divergence + pixel
        criteria.set("divergence", divergence)
        criteria.set("pixel", pixel)
        criteria.set("total", total)
        return(criteria)

    def getGeneration(self, number: int) -> torch.Tensor:
        # generation = tensordict.TensorDict(device=self.device)
        # getReconstruction
        shape = (number, 8, 4, 4)
        sample = torch.randn(*shape, device=self.device)
        generation = self.getReconstruction(sample)
        # route = getattr(self.layer, 'decode')
        # generation = route(noise)['sample']
        return(generation)

    def getRepresentation(self, image: torch.Tensor) -> torch.Tensor:
        distribution = self.getDistribution(image)
        representation = distribution['mean'].flatten(1, -1)
        return(representation)

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
