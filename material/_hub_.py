import torch
import os
import torchvision
import torchcodec
import functools
import PIL.Image
import tensordict

def getCollation(queue: list) -> tensordict.TensorDict:
    bundle = {
        'image': []
    }
    iteration = queue
    for item in iteration:
        path = item
        getComposition = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((64, 64)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    [0.5, 0.5, 0.5], 
                    [0.5, 0.5, 0.5]
                )
            ]
        )
        image = getComposition(
            PIL.Image.open(path)
        )
        bundle['image'] += [image]
        continue
    _ = iteration
    source = {
        'image': torch.stack(bundle['image'], dim=0)
    }
    size = len(queue)
    collation = tensordict.TensorDict(
        source = source,
        batch_size = size
    ).detach()
    return(collation)

class Unit(torch.utils.data.Dataset):

    def __init__(self, queue: list) -> None:
        self.queue = queue
        return
    
    def getLength(self) -> int:
        length = len(self.queue)
        return(length)

    def getItem(self, index: int) -> tuple:
        item = self.queue[index]
        return(item)

    __len__ = getLength
    __getitem__ = getItem
    pass

class Document:

    def __init__(self, path: str) -> None:
        self.path = path
        return

    def getQueue(self) -> list:
        folder = os.path.dirname(self.path)
        paper = open(self.path, 'r')
        queue = []
        iteration = paper.readlines()
        for item in iteration:
            path = item.replace("\n", "")
            queue += [os.path.join(folder, path)]
            continue
        _ = iteration
        paper.close()
        return(queue)

    pass

class Hub:

    def __init__(self) -> None:
        return

    def getData(self, number: int) -> torch.utils.data.DataLoader:
        name = 'data.txt'
        path = os.path.join(self.folder, name)
        queue = Document(path).getQueue()
        unit = Unit(queue)
        data = torch.utils.data.DataLoader(
            dataset=unit,
            batch_size=number,
            shuffle=True,
            collate_fn=getCollation, #functools.partial(getCollation, device=self.device),
            drop_last=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )
        return(data)

    def getValidation(
        self, 
        number: int, 
        reproducibility: bool
    ) -> torch.utils.data.DataLoader:
        name = 'validation.txt'
        path = os.path.join(self.folder, name)
        queue = Document(path).getQueue()
        unit = Unit(queue)
        validation = torch.utils.data.DataLoader(
            dataset=unit,
            batch_size=number,
            shuffle=not reproducibility,
            collate_fn=getCollation, #functools.partial(getCollation, device=self.device),
            drop_last=not reproducibility,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True
        )
        return(validation)
    
    def getTest(
        self, 
        number: int,
        reproducibility: bool
    ) -> torch.utils.data.DataLoader:
        name = 'test.txt'
        path = os.path.join(self.folder, name)
        queue = Document(path).getQueue()
        unit = Unit(queue)
        test = torch.utils.data.DataLoader(
            dataset=unit,
            batch_size=number,
            shuffle=not reproducibility,
            collate_fn=getCollation, #functools.partial(getCollation, device=self.device),
            drop_last=not reproducibility,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True
        )
        return(test)

    def getBatch(self, number: int) -> tensordict.TensorDict:
        name = 'data.txt'
        path = os.path.join(self.folder, name)
        queue = Document(path).getQueue()
        unit = Unit(queue)
        data = torch.utils.data.DataLoader(
            dataset=unit,
            batch_size=number,
            shuffle=True,
            collate_fn=getCollation, #functools.partial(getCollation, device=self.device),
            drop_last=True
        )
        batch = next(iter(data))
        return(batch)

    folder = 'material/storage'
    pass
