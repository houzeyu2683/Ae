import torch
import os
import itertools
import safetensors.torch
import bitsandbytes
import visualization
import tensordict
import tqdm

class Framework:

    def __init__(
        self, 
        model: torch.nn.Module, 
        device: str,
        history: str
    ) -> None:
        self.model = model
        self.device = device
        self.history = history
        return

    def saveModel(self, path: str) -> bool:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model, path)
        return(True)

    def saveWeight(self, path: str) -> bool:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        safetensors.torch.save_file(self.model.state_dict(), path)
        return(True)

    def loadModel(self, path: str) -> bool:
        self.model = torch.load(path)
        return(True)

    def loadWeight(self, path: str) -> bool:
        state_dict = safetensors.torch.load_file(path)
        self.model.load_state_dict(state_dict)
        return(True)

    # def getCriteria(
    #     self, 
    #     estimation: tensordict.TensorDict,
    #     scale: float
    # ) -> tensordict.TensorDict:
    #     # 
    #     mean = estimation['distribution', 'mean']
    #     variance = estimation['distribution','variance']
    #     distance = -0.5 * torch.sum(
    #         1 + variance.log() - mean.pow(2) - variance,
    #         dim=(1, 2, 3)   # sum over (C, H, W)
    #     )
    #     divergence = scale * distance.mean()
    #     # pixel = torch.sum((image-reconstruction)**2, dim=[1,2,3]).mean()
    #     image = estimation['image']
    #     reconstruction = estimation['reconstruction']
    #     error = torch.sum(
    #         (image-reconstruction).pow(2), 
    #         dim=[1,2,3]
    #     ).mean()
    #     total = divergence + error
    #     criteria = tensordict.TensorDict(device=self.device)
    #     criteria.set("divergence", divergence)
    #     criteria.set("mean squared error", error)
    #     criteria.set("total", total)
    #     return(criteria)

    def fitWeight(
        self, 
        data: torch.utils.data.DataLoader,
        snapshot: int,
        # history: str,
        total: int,
        accumulation: int,
        validation: torch.utils.data.DataLoader
    ) -> bool:
        # optimization
        rate = 1e-4
        optimization = bitsandbytes.optim.AdamW(
            self.model.parameters(), lr=rate
        )
        optimization.zero_grad()
        # dashboard
        dashboard = visualization.Dashboard(self.history)
        dashboard.openSession()
        #
        self.saveModel(
            os.path.join(self.history, f'weight/model.pt')
        )
        #
        self.model.train()
        gradient = torch.amp.GradScaler()
        number = 0
        while(True):
            # iteration = enumerate(itertools.cycle(data), 0)
            iteration = data
            for batch in tqdm.tqdm(iteration):
                scale = 1.0#0.001 #1e-2#1.0#min(1.0, (number/10000)*0.1)
                with torch.amp.autocast(self.device):
                    getCriteria = getattr(self.model, 'getCriteria')
                    criteria = getCriteria(batch, scale)
                    pass
                loss = criteria['total'] / accumulation  # 分攤梯度
                assert isinstance(loss, torch.Tensor)
                gradient.scale(loss).backward()
                if((number+1)%accumulation==0):
                    gradient.step(optimization)
                    gradient.update()
                    optimization.zero_grad()
                    pass
                dashboard.insertElement(
                    'Data/Loss(Total)', 
                    loss * accumulation, 
                    number
                )
                dashboard.insertElement(
                    'Data/Loss(Divergence)', 
                    criteria['divergence'], 
                    number
                )
                dashboard.insertElement(
                    'Data/Loss(Mean Squared Error)', 
                    criteria['mean squared error'], 
                    number
                )
                if(number%snapshot==0):
                    checkpoint = os.path.join(
                        self.history, 
                        'weight',
                        f'{number}.pt'
                    )
                    self.saveWeight(path=checkpoint)
                    # 可视化重建质量
                    # if(validation):
                    self.model.eval()
                    with torch.no_grad():
                        # Validation
                        batch = next(iter(validation))
                        estimation = self.model(batch)
                        image = batch['image']
                        reconstruction = estimation['reconstruction']
                        dashboard.insertPicture(
                            'Validation', 
                            torch.cat([image, reconstruction], dim=0), 
                            number
                        )
                        # Test
                        getGeneration = getattr(self.model, 'getGeneration')
                        generation = getGeneration(64)
                        dashboard.insertPicture(
                            'Generation', 
                            generation,
                            number
                        )
                        pass
                    self.model.train()
                    pass
                number += 1
                termination = (total==number)
                if(termination): break
                continue
            _ = iteration
            if(termination): break
            continue
        dashboard.closeSession()
        return(True)

    @torch.no_grad()
    def getReconstruction(
        self, 
        image: torch.Tensor,
    ) -> torch.Tensor:
        _, _, reconstruction = self.model(image)
        return(reconstruction)

    pass

