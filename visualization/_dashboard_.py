import os
import torch
import typing
import torchvision
import torch.utils.tensorboard

class Dashboard:

    def __init__(self, directory: str) -> None:
        self.directory = directory
        return

    def openSession(self) -> bool:
        os.makedirs(self.directory, exist_ok=True)
        self.session = torch.utils.tensorboard.SummaryWriter(self.directory)
        return(True)

    def closeSession(self) -> bool:
        self.session.close()
        return(True)

    def insertElement(self, tag: str, value: torch.Tensor, number: int) -> bool:
        self.session.add_scalar(tag, value, number)
        return(True)

    # def insertGrapgh(
    #     self, model: torch.nn.Module, variable: typing.Any
    # ) -> bool:
    #     self.session.add_graph(model, variable)
    #     return(True)
    
    def insertPicture(
        self, tag: str, image: torch.Tensor, number: int
    ) -> bool:
        grid = torchvision.utils.make_grid(image)
        self.session.add_image(tag, grid, number)
        return(True)

    pass
