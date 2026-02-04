import torch
import os
import safetensors.torch
import bitsandbytes
import visualization
import tqdm
import torchvision
import tensordict

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

    def saveWeight(self, path: str) -> bool:
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
        #     T_mult=2,
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
                with torch.amp.autocast(self.device):
                    criteria = self.model(batch)
                    pass
                loss = torch.div(criteria['total'], accumulation)
                gradient.scale(loss).backward()
                if(number%accumulation==0):
                    gradient.step(optimization)
                    gradient.update()
                    optimization.zero_grad()
                    pass
                element = {
                    'Total': criteria['total'],
                    'Commitment': criteria['commitment'],
                    'Pixel': criteria['pixel'],
                    # 'Brightness': criteria['brightness']
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
                    criteria = self.model(batch)
                    pass
                self.model.train()
                element = {
                    'Total': criteria['total'],
                    'Commitment': criteria['commitment'],
                    'Pixel': criteria['pixel'],
                    # 'Brightness': criteria['brightness']
                }
                dashboard.insertStatistic(
                    'Loss/Validation',
                    element,
                    number
                )
                # Snapshot
                if(number==1 or (number%snapshot)==0):
                    checkpoint = os.path.join(
                        self.history, 
                        'weight',
                        f'{number}.pt'
                    )
                    self.saveWeight(path=checkpoint)
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

    @torch.no_grad()
    def makeInference(self, batch: tensordict.TensorDict) -> bool:
        batch = batch.to(self.device, non_blocking=True)
        self.model.eval()
        getRepresentation = getattr(
            self.model, 'getRepresentation'
        )
        getReconstruction = getattr(
            self.model, 'getReconstruction'
        )
        image = batch['image']
        representation = getRepresentation(image)
        quantization = representation['quantization']
        reconstruction = getReconstruction(quantization)
        inference = {
            'image': image,
            'reconstruction': reconstruction
        }
        # inference = torch.cat([image, reconstruction], dim=0)
        self.inference = inference
        return(True)

    def saveInference(self, archive: str, comparison: bool) -> bool:
        tag = 'inference'
        folder = os.path.join(self.history, tag)
        os.makedirs(folder, exist_ok=True)
        if(comparison):
            image = self.inference['image']
            reconstruction = self.inference['reconstruction']
            inference = torch.cat([image, reconstruction], dim=0)
            pass
        else:
            inference = self.inference['reconstruction']
            pass
        torchvision.utils.save_image(
            inference,
            os.path.join(folder, archive),
            normalize=True,
            value_range=(-1, 1)
        )
        return(True)

    pass



                    #
                    # self.model.eval()
                    # with torch.no_grad():
                    #     getRepresentation = getattr(
                    #         self.model, 'getRepresentation'
                    #     )
                    #     getReconstruction = getattr(
                    #         self.model, 'getReconstruction'
                    #     )
                    #     batch = next(iter(validation))
                    #     image = batch['image']
                    #     representation = getRepresentation(image)
                    #     quantization = representation['quantization']
                    #     reconstruction = getReconstruction(quantization)
                    #     pass
                    # image = getattr(image, 'to')(self.device)
                    # overview = torch.cat([image, reconstruction], dim=0)
                    # dashboard.insertPicture(
                    #     'Validation/Overview', 
                    #     overview, 
                    #     number
                    # )
                    # self.model.train()
