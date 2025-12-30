import torch
import diffusers
import tensordict

class Sparrow(torch.nn.Module):

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
            down_block_types=["DownEncoderBlock2D"] * 5,
            up_block_types=["UpDecoderBlock2D"] * 5
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
        getReparameterization = getattr(space['latent_dist'], 'sample')
        reparameterization = getReparameterization()
        distribution.set("mean", mean)
        distribution.set("variance", variance)
        distribution.set("reparameterization", reparameterization)
        return(distribution)
    
    def getReconstruction(
        self, 
        reparameterization: torch.Tensor
    ) -> torch.Tensor:
        route = getattr(self.layer, 'decode')
        node = route(reparameterization)
        reconstruction = getattr(node, 'sample')
        return(reconstruction)

    def getEstimation(
        self, 
        batch: tensordict.TensorDict
    ) -> tensordict.TensorDict:
        estimation = tensordict.TensorDict(device=self.device)
        image = batch['image']
        distribution = self.getDistribution(image)
        reparameterization = distribution['reparameterization']
        reconstruction = self.getReconstruction(reparameterization)
        estimation.set('distribution', distribution)
        estimation.set('reconstruction', reconstruction)
        return(estimation)

    def getCriteria(
        self, 
        batch: tensordict.TensorDict,
        scale: float
    ) -> tensordict.TensorDict:
        criteria = tensordict.TensorDict(device=self.device)
        estimation = self.getEstimation(batch)
        # image = batch['image']
        mean = estimation['distribution', 'mean']
        variance = estimation['distribution','variance']
        distance = -0.5 * torch.sum(
            1 + variance.log() - mean.pow(2) - variance,
            dim=(1, 2, 3)   # sum over (C, H, W)
        )
        divergence = scale * distance.mean()
        # pixel = torch.sum((image-reconstruction)**2, dim=[1,2,3]).mean()
        image = batch['image']
        reconstruction = estimation['reconstruction']
        error = torch.sum(
            (image-reconstruction).pow(2), 
            dim=[1,2,3]
        ).mean()
        total = divergence + error
        criteria.set("divergence", divergence)
        criteria.set("mean squared error", error)
        criteria.set("total", total)
        return(criteria)

    def getGeneration(self, number: int) -> torch.Tensor:
        # generation = tensordict.TensorDict(device=self.device)
        shape = (number, 8, 4, 4)
        noise = torch.randn(*shape, device=self.device)
        route = getattr(self.layer, 'decode')
        generation = route(noise)['sample']
        return(generation)

    forward = getEstimation
    pass


