"""
Copied and modified from [MOTR repo](https://github.com/megvii-research/MOTR/blob/main/datasets/detmot.py)
Cannot be used currently.
"""

from pathlib import Path
from PIL import Image
from typing import Union, Callable, Iterable, Literal
from torch.utils.data import Dataset

import numpy as np
import os, configparser, torch
import torchvision.transforms.functional as F

from ..utils.api import GroundTruth
from . import transforms as T


# _mot_train_videos = [f"MOT17-{i}-DPM" for i in ["02", "04", "05", "09", "11", "13"]]
# _mot_val_videos = ["MOT17-10-DPM"]

class MOT17DynamicDataset(Dataset):
        """
        MOT17 benchmark sample generator. By `Dynamic`, means a video clip is loaded for multi-object tracking models.
        """
        def __init__(self, 
                     root: str, 
                     num_frames: int = 4, 
                     num_iters: Union[int, Iterable] = 1, 
                     max_sample_interval: int = 1,
                     min_sample_interval: int = 1,
                     sample_interval_mode: Literal["fixed", "random"] = "random",
                     update_epochs: Union[int, Iterable] = None, 
                     mode: Literal["train", "val", "test"] = "train", 
                     val_ratio: float = 0.3,
                     transforms: Callable = None) -> None:
            super().__init__()
            self.root = root
            self.num_frames = num_frames
            self.num_iters = [num_iters] if not isinstance(num_iters, Iterable) else num_iters
            self.max_sample_interval = max_sample_interval
            self.min_sample_interval = min_sample_interval
            self.sample_interval_mode = sample_interval_mode
            self.update_epochs = [np.inf] if not update_epochs else update_epochs
            assert max_sample_interval >= min_sample_interval
            self.mode = mode
            self.val_ratio = val_ratio
            self.tranforms = transforms

            self.dir = os.path.join(root, "train" if mode in ["train", "val"] else mode)
            self.video_names = sorted([
                video for video in os.listdir(self.dir)
                if not video == ".DS_Store" and video.endswith("DPM") # MOT16
            ])
            self.video_infos: dict[str, dict] = {}

            self.num_images: int = 0
            self.samples: list[tuple[str, np.ndarray]] = []

            self.current_epoch: int = 0
            self.current_num_iter: int = self.num_iters[0]

            self.sample_num = num_frames + self.current_num_iter - 1
            self._cur_index: int = 0

            self._register_videos()
            self.reset()

        def __len__(self) -> int:
            return len(self.samples)
        
        def __getitem__(self, index: int) -> GroundTruth:
            video_name, frame_indices = self.samples[index]

            anno_path = self.video_infos[video_name]["anno_path"]
            img_h, img_w = [self.video_infos[video_name][name] for name in ["imHeight", "imWidth"]]
            img_paths = [self.video_infos[video_name]["img_paths"][i] for i in frame_indices]
            imgs = [Image.open(path) for path in img_paths]

            # FIXME: pseudo images， used for loader testing
            # imgs = [
            #     Image.fromarray(
            #         np.random.randint(0, 256, size=[img_h, img_w, 3]).astype(np.uint8)
            #     )
            #     for _ in range(len(img_paths))
            # ]

            annos = np.loadtxt(anno_path, delimiter=",")
            annos = annos[annos[:, 0].argsort()] # sort by frame FIXME: not necessary
            anno_list: list[dict] = []

            for index in frame_indices:
                anno_dict: dict[str, Union[torch.Tensor, int]] = {}
                frame_annos = annos[np.logical_and(annos[:, 0] == index, annos[:, 6] == 1)]
                
                # x_min, y_min, w, h -> x_min, y_min, x_max, y_max
                # T.Normalize() will convert to cxcywh and normalize boxes
                frame_annos[:, 4] = (frame_annos[:, 2] + frame_annos[:, 4]).clip(0., img_w)
                frame_annos[:, 5] = (frame_annos[:, 3] + frame_annos[:, 5]).clip(0., img_h)
                # frame_annos[:, 4] /= img_w
                # frame_annos[:, 5] /= img_h

                frame_annos = torch.from_numpy(frame_annos).float()
                anno_dict["labels"] = torch.zeros(frame_annos.size(0)).long() # class id: 0
                anno_dict["ids"] = frame_annos[:, 1].long()
                anno_dict["boxes"] = frame_annos[:, 2:6]
                anno_dict["area"] = frame_annos[:, 4] * frame_annos[:, 5]
                anno_dict['org_size'] = torch.as_tensor([img_h, img_w])

                # if self.mode in ["val", "test"]:
                #     _num_iter = len(frame_indices) - self.num_frames + 1
                # else:
                #     _num_iter = self.current_num_iter
                anno_dict["num_iter"] = len(frame_indices) - self.num_frames + 1

                anno_list.append(anno_dict)

            if self.tranforms:
                imgs, anno_list = self.tranforms(imgs, anno_list)
                imgs = torch.stack(imgs, dim=0)
            else:
                imgs = torch.stack([F.to_tensor(img) for img in imgs], dim=0)

            return GroundTruth(imgs, anno_list)

        def _register_videos(self):
            for video_name in self.video_names:
                video_path = os.path.join(self.dir, video_name)
                name, info = self._parse_seq_info(video_path)

                # img_paths = sorted([os.path.join(info["imDir"], name) for name in os.listdir(info["imDir"])])
                img_paths = [f"{info['imDir']}/{i:06d}{info['imExt']}" for i in range(info["seqLength"])]
                slice_index = int(len(img_paths) * (1 - self.val_ratio))
                if self.mode == "train":
                    img_paths = img_paths[:slice_index]
                elif self.mode == "val":
                    img_paths = img_paths[slice_index:]

                info["anno_path"] = os.path.join(video_path, "gt", "gt.txt")
                info["img_paths"] = img_paths

                self.video_infos[name] = info
                self.num_images += len(img_paths)

        @staticmethod
        def _parse_seq_info(ini_path: str):
            config = configparser.ConfigParser()
            config.read(os.path.join(ini_path, "seqinfo.ini"), encoding="utf-8-sig")

            try:
                seq_info = {
                    'imDir': os.path.join(ini_path, config.get('Sequence', 'imDir')),
                    'frameRate': config.getfloat('Sequence', 'frameRate'),
                    'seqLength': config.getint('Sequence', 'seqLength'),
                    'imWidth': config.getint('Sequence', 'imWidth'),
                    'imHeight': config.getint('Sequence', 'imHeight'),
                    'imExt': config.get('Sequence', 'imExt')
                }
            except Exception as exc:
                print("error occured at file `{}`:".format(ini_path))
                raise exc
            
            return config.get('Sequence', 'name'), seq_info
        
        def _sample_imgs_for_one_sample(self):
            self.samples.clear()
            start_index: int = np.random.randint(self.min_sample_interval, self.max_sample_interval)

            for video_name, video_info in self.video_infos.items():
                num_imgs = len(video_info["img_paths"])

                if self.sample_interval_mode == "fixed":
                    indices = np.arange(num_imgs, dtype=np.int64)
                    indices = indices[::self.max_sample_interval] # fixed interval sampling
                else: # random interval
                    intervals = np.random.randint(self.min_sample_interval, self.max_sample_interval + 1, [num_imgs]) # get intervals
                    indices = np.concatenate([[start_index], intervals]).cumsum() # cumsum to get indices
                    indices = indices[indices < num_imgs] # mask out out-of-range indices

                # make sample divisible by num_images_per_sample
                residual = len(indices) % self.sample_num
                if residual > 0:
                    indices = indices[:-residual]

                
                if self.mode in ["val", "test"]:
                    self.samples.append((video_name, indices.flatten()))
                else:
                    indices = indices.reshape(-1, self.sample_num)
                    for index in indices:
                        self.samples.append((video_name, index))

        def reset(self):
            self.current_epoch = 0
            self.current_epoch: int = 0
            self.current_num_iter: int = self.num_iters[0]

            self.sample_num = self.num_frames + self.current_num_iter - 1
            self._cur_index: int = 0

            self._sample_imgs_for_one_sample()

        def update_state(self):
            if self._cur_index < len(self.update_epochs) and self.current_epoch >= self.update_epochs[self._cur_index]:
                self._cur_index += 1
                self.current_num_iter = self.num_iters[self._cur_index]

                self.sample_num = self.num_frames + self.current_num_iter - 1

        def set_epoch(self, epoch: int):
            self.current_epoch = epoch
            self.update_state()
            self._sample_imgs_for_one_sample()

        def step_epoch(self):
            self.set_epoch(self.current_epoch + 1)


def collate_fn_mot(batch: list[GroundTruth]):
    return batch


def make_detmot_transforms(mode: Literal["train", "val"], args=None):
    max_size = 1333 if args is None else args.max_size
    val_width = 800 if args is None else args.val_width
    normalize = T.MotCompose([
        T.MotToTensor(),
        T.MotNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    # scales = [480 - 32]

    if mode == 'train':
        color_transforms = []
        scale_transforms = [
            T.MotRandomHorizontalFlip(),
            T.MotRandomResize(scales, max_size=max_size),
            normalize,
        ]

        return T.MotCompose(color_transforms + scale_transforms)

    if mode == 'val':
        return T.MotCompose([
            T.MotRandomResize([val_width], max_size=max_size),
            normalize,
        ])

    raise ValueError(f'unknown {mode}')


def build_mot_dataset(mode: Literal["train", "val"], args):
    root = Path(args.root)
    assert root.exists(), f'provided MOT path {root} does not exist'
    transforms = make_detmot_transforms(mode, args)

    dataset = MOT17DynamicDataset(root, args.num_frames, args.num_iters, 
                                  args.max_sample_interval, args.min_sample_interval,
                                  args.sample_mode, args.resample_epochs, mode=mode, val_ratio=args.val_ratio,
                                  transforms=transforms)

    return dataset


__all__ = [
    "MOT17DynamicDataset",
    "collate_fn_mot",
    "make_detmot_transforms",
    "build_mot_dataset"
]