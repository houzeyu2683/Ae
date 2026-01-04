import torch
import os
import safetensors.torch
import bitsandbytes
import visualization
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

    def saveWeight(self, path: str) -> bool:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        safetensors.torch.save_file(self.model.state_dict(), path)
        return(True)

    def loadWeight(self, path: str) -> bool:
        state_dict = safetensors.torch.load_file(path)
        self.model.load_state_dict(state_dict)
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
                with torch.amp.autocast(self.device):
                    criteria = self.model(batch)
                    pass
                loss = torch.div(criteria['total'], accumulation)
                gradient.scale(loss).backward()
                if((number)%accumulation==0):
                    gradient.step(optimization)
                    # schedule.step()
                    gradient.update()
                    optimization.zero_grad()
                    pass
                element = {
                    'Total': criteria['total'],
                    'Commitment': criteria['commitment'],
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
                    # image = batch['image']
                    criteria = self.model(batch)
                    element = {
                        'Total': criteria['total'],
                        'Commitment': criteria['commitment'],
                        'Pixel': criteria['pixel']
                    }
                    dashboard.insertStatistic(
                        'Loss/Validation',
                        element,
                        number
                    )
                    pass
                self.model.train()
                # Snapshot
                if((number==1) or (number)%snapshot==0):
                    checkpoint = os.path.join(
                        self.history, 
                        'weight',
                        f'{number}.pt'
                    )
                    self.saveWeight(path=checkpoint)
                    self.model.eval()
                    with torch.no_grad():
                        getRepresentation = getattr(
                            self.model, 'getRepresentation'
                        )
                        getReconstruction = getattr(
                            self.model, 'getReconstruction'
                        )
                        batch = next(iter(validation))

                        # criteria = self.model(batch)
                        # reconstruction_2 = criteria['reconstruction']

                        image = batch['image']
                        representation = getRepresentation(image)
                        quantization = representation['quantization']
                        reconstruction = getReconstruction(quantization)

                        # reconstruction==reconstruction_2

                        overview = torch.cat([image, reconstruction], dim=0)
                        dashboard.insertPicture(
                            'Validation/Overview', 
                            overview, 
                            number
                        )
                        pass
                    self.model.train()
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

    pass

