import torch
import os
import safetensors.torch
import bitsandbytes
import visualization
import tqdm
import tensordict
import torchvision

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

    def saveCheckpoint(self, path: str) -> bool:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        safetensors.torch.save_file(self.model.state_dict(), path)
        return(True)

    def getMemory(self) -> str:
        freeness, total = torch.cuda.mem_get_info()
        occupancy = total - freeness
        value = round((occupancy / total) * 100, 2)
        memory = f"{value}%"
        return(memory)

    def fitWeight(
        self, 
        data: torch.utils.data.DataLoader,
        snapshot: int,
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
        #
        # schedule = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #     optimization,
        #     T_0=10000,
        #     T_mult=1,
        #     eta_min=1e-6
        # )
        # dashboard
        dashboard = visualization.Dashboard(self.history)
        dashboard.openSession()
        #
        self.model.train()
        gradient = torch.amp.GradScaler()
        number = 1
        while(True):
            iteration = tqdm.tqdm(data)
            for batch in iteration:
                memory = self.getMemory()
                iteration.set_postfix({"Memory": memory})
                # Data
                # period = number//(5*len(data))
                # scale = min(1e-5, 1e-6*period)
                with torch.amp.autocast(self.device):
                    # criteria = self.model(batch, scale)
                    criteria = self.model(batch)
                    pass
                loss = torch.div(criteria['total'], accumulation)
                gradient.scale(loss).backward()
                if(number%accumulation==0):
                    gradient.step(optimization)
                    # schedule.step()
                    gradient.update()
                    optimization.zero_grad()
                    pass
                element = {
                    'Total': criteria['total'],
                    # 'Divergence': criteria['divergence'],
                    'Pixel': criteria['pixel']
                }
                dashboard.insertStatistic(
                    'Loss/Data',
                    element,
                    number
                )
                # Validation
                self.model.eval()
                with torch.no_grad():
                    batch = next(iter(validation))
                    # criteria = self.model(batch, scale)
                    criteria = self.model(batch)
                    pass
                self.model.train()
                element = {
                    'Total': criteria['total'],
                    # 'Divergence': criteria['divergence'],
                    'Pixel': criteria['pixel']
                }
                dashboard.insertStatistic(
                    'Loss/Validation',
                    element,
                    number
                )
                # Snapshot
                if(number==1 or number%snapshot==0):
                    path = os.path.join(
                        self.history, 
                        'checkpoint',
                        f'{number}.pt'
                    )
                    self.saveCheckpoint(path)
                    pass
                number += 1
                termination = False if(total==-1) else (total<number)
                if(termination): break
                continue
            _ = iteration
            if(termination): break
            continue
        dashboard.closeSession()
        return(True)

    def saveWeight(self, checkpoint: list) -> bool:
        # folder = os.path.join(self.history, 'weight')
        index = checkpoint.pop(0)
        path = os.path.join(self.history, 'checkpoint', index)
        aggregate = {}
        iteration = safetensors.torch.load_file(path).items()
        for key, value in iteration:
            aggregate.update({key: value.clone()})
            continue
        _ = iteration
        #
        iteration = checkpoint
        for index in iteration:
            path = os.path.join(self.history, 'checkpoint', index)
            state = safetensors.torch.load_file(path)
            for key in aggregate: aggregate[key] += state[key]
            continue
        _ = iteration
        size = len(checkpoint) + 1
        for key in aggregate: aggregate[key] /= size
        structure = self.model.state_dict()
        assert aggregate.keys() == structure.keys()
        weight = aggregate
        path = os.path.join(self.history, 'weight.pt')
        safetensors.torch.save_file(weight, path)
        return(True)
    
    pass
    # @torch.no_grad()
    # def makeComparison(self, batch: tensordict.TensorDict) -> bool:
    #     batch = batch.to(self.device, non_blocking=True)
    #     self.model.eval()
    #     # with torch.no_grad():
    #     image = batch['image']
    #     getCompression = getattr(self.model, 'getCompression')
    #     getReconstruction = getattr(self.model, 'getReconstruction')
    #     compression = getCompression(image)
    #     # sample = distribution['sample']
    #     reconstruction = getReconstruction(compression)
    #         # pass
    #     comparison = torch.cat([image, reconstruction], dim=0)
    #     self.comparison = comparison
    #     return(True)

    # def saveComparison(self, archive: str) -> bool:
    #     tag = 'comparison'
    #     folder = os.path.join(self.history, tag)
    #     os.makedirs(folder, exist_ok=True)
    #     torchvision.utils.save_image(
    #         self.comparison,
    #         os.path.join(folder, archive),
    #         normalize=True,
    #         value_range=(-1, 1)
    #     )
    #     return(True)
    
    # @torch.no_grad()
    # def makePerspective(self, number: int) -> bool:
    #     shape = (number, 8, 4, 4)
    #     self.model.eval()
    #     # with torch.no_grad():
    #     sample = torch.randn(*shape, device=self.device)
    #     getReconstruction = getattr(self.model, 'getReconstruction')
    #     perspective = getReconstruction(sample)
    #         # pass
    #     self.perspective = perspective
    #     return(True)

    # def savePerspective(self, archive: str) -> bool:
    #     tag = 'perspective'
    #     folder = os.path.join(self.history, tag)
    #     os.makedirs(folder, exist_ok=True)
    #     torchvision.utils.save_image(
    #         self.perspective,
    #         os.path.join(folder, archive),
    #         normalize=True,
    #         value_range=(-1, 1)
    #     )
    #     return(True)

    # def makeInference(self, batch: tensordict.TensorDict) -> bool:
    #     image = batch['image'].to(self.device, non_blocking=True)
    #     getDistribution = getattr(
    #         self.model, 'getDistribution'
    #     )
    #     getReconstruction = getattr(
    #         self.model, 'getReconstruction'
    #     )
    #     self.model.eval()
    #     with torch.no_grad():
    #         distribution = getDistribution(image)
    #         sample = distribution['sample']
    #         reconstruction = getReconstruction(sample)
    #         pass
    #     # quantization = representation['quantization']
    #     # reconstruction = getReconstruction(quantization)
    #     inference = torch.cat([image, reconstruction], dim=0)
    #     self.inference = inference
    #     return(True)

    # def saveInference(self, name: str) -> bool:
    #     tag = 'inference'
    #     folder = os.path.join(self.history, tag)
    #     os.makedirs(folder, exist_ok=True)
    #     torchvision.utils.save_image(
    #         self.inference,
    #         os.path.join(folder, f'{name}.jpg'),
    #         normalize=True,
    #         value_range=(-1, 1)
    #     )
    #     return(True)



                    # self.model.eval()
                    # with torch.no_grad():
                    #     getDistribution = getattr(
                    #         self.model, 
                    #         'getDistribution'
                    #     )
                    #     getReconstruction = getattr(
                    #         self.model, 
                    #         'getReconstruction'
                    #     )
                    #     getGeneration = getattr(
                    #         self.model, 
                    #         'getGeneration'
                    #     )
                    #     batch = next(iter(validation))
                    #     image = batch['image']
                    #     distribution = getDistribution(image)
                    #     sample = distribution['sample']
                    #     reconstruction = getReconstruction(sample)
                    #     generation = getGeneration(64)
                    #     pass
                    # self.model.train()
                    # image = getattr(image, 'to')(self.device)
                    # overview = torch.cat(
                    #     [image, reconstruction], 
                    #     dim=0
                    # )
                    # dashboard.insertPicture(
                    #     'Validation/Overview', 
                    #     overview, 
                    #     number
                    # )
                    # dashboard.insertPicture(
                    #     'Validation/Generation', 
                    #     generation,
                    #     number
                    # )