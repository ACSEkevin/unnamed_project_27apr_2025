import torch

from torch.utils.data import Dataset

from ..utils.api import GroundTruth


class PseudoDataSet(Dataset):
    def __init__(self, num_images: int, num_classes: int = 91, num_frames: int = 4) -> None:
        super().__init__()
        self.num_frames = num_frames
        self.num_images = num_images

        self.samples = [
            GroundTruth(
                torch.randn([self.num_images, 3, 480, 600]).float(),
                [dict(
                    labels=torch.randint(0, num_classes, [num_objs]),
                    boxes=torch.randn([num_objs, 4]).float().sigmoid(),
                    ids=torch.randperm(num_objs + 10)[:num_objs],
                    areas=torch.randn([num_objs]).float().sigmoid(), 
                    num_iter=self.num_images - self.num_frames + 1,
                ) for num_objs in torch.randint(1, 30, [self.num_images]).tolist()]
            )
            for _ in range(16)
        ]

    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, index: int) -> GroundTruth:
        return self.samples[index]
    
    def set_iter_num(self, num_iter: int):
        """
        method as a placeholder for aligning dataset interface 
        """
        ...


def collate_fn_pseudo_dataset(batch: list[GroundTruth]):
    return batch
        

__all__ = [
     "PseudoDataSet",
     "collate_fn_pseudo_dataset"
]