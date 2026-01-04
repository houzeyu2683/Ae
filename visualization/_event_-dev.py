import tensorboard.backend.event_processing.event_accumulator
import io
import PIL.Image

ea = tensorboard.backend.event_processing.event_accumulator.EventAccumulator("./log/robin-01030705/", size_guidance={"images": 1e8})
ea.Reload()
lock = ea.Images("Validation/Overview")

class Event:

    def __init__(self, folder: str):
        self.folder = folder
        return

    def readArchive(self) -> bool:
        node = tensorboard.backend.event_processing.event_accumulator
        archive = node.EventAccumulator(
            self.folder, size_guidance=self.guidance
        )
        archive.Reload()
        self.archive = archive
        return(True)
    
    guidance = {
        "images": 1000
    }

    pass