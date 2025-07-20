from torch import Tensor
from typing import Union, Iterable, Callable

import torch


class GroundTruth:
    """
    A `GroundTruth` defines a `Dataset` sample structure / output interface.
    """
    def __init__(self, images: Union[Tensor, list[Tensor]], annos: list[dict[str, Union[Tensor, int]]]) -> None:
        """
        Args:
        ---
            - images: Tensor with a shape of [T, 3, H, W]
            - labels: annotation infos which must contain keys:
                - `labels`: category info, shape [N_obj]
                - `boxes`: normalized box with a format of [cx, cy, w, h], shape [N_obj, 4]
                - `ids`: object instance id, if cannot read id from annotations, gives -1, shape [N_obj]
                - `area`: bounding box area, if 0, shape[N_obj]
                - `num_iter`: number of iterations in this round.
        """

        self.images = self._validate_imges_type(images)
        self.num_images = len(images)
        self.annos = annos
        self.num_iters: int = annos[0]["num_iter"]
        self.num_frames = self.num_images - self.num_iters + 1

        self.num_objects = self._create_num_objects()
        self.id_to_label = self._create_id_to_label_map()
        self.appearence, self.id_to_index = self._create_appearence_info()

        for anno in self.annos:
            for key in anno.keys():
                if key in ["boxes", "num_iter"]:
                    continue
                    
                anno[key] = anno[key].unsqueeze(1)

        assert self.num_images == len(self.annos)

    def __len__(self) -> int:
        return self.images.size(0)
    
    def __getitem__(self, index: int):
        return self.images[index], self.annos[index]
    
    def __repr__(self) -> str:
        return "{}(num_images={}, num_frames={}, num_instances={})".format(
            self.__class__.__name__, self.num_images, self.num_frames, len(self.id_to_index)
        )
    
    @staticmethod
    def _validate_imges_type(images: Union[Tensor, list[Tensor]]):
        assert isinstance(images, (Tensor, list)), type(images)
        if isinstance(images, list):
            for img in images:
                assert isinstance(img, Tensor), type(img)

        return images
    
    def _create_num_objects(self) -> list[int]:
        return [anno["labels"].numel() for anno in self.annos]
    
    def _create_appearence_info(self):
        id_sequence = [anno["ids"].flatten() for anno in self.annos]
        all_ids: Tensor = torch.cat(id_sequence).cpu().unique()
        num_ids = len(all_ids)
        
        # id to index map
        id_to_index = {id.item(): idx for idx, id in enumerate(all_ids)}
        appearance = torch.zeros(num_ids, self.num_images, dtype=torch.bool).cpu()
        
        for frame_idx, frame_ids in enumerate(id_sequence):
            if len(frame_ids) == 0: # no objects
                continue
                
            # get indices of ids in one frame
            indices = torch.tensor([id_to_index[id.item()] for id in frame_ids]).cpu().long()
            appearance[indices, frame_idx] = True # set appeared to indices
        
        return appearance, id_to_index
    
    def _create_id_to_label_map(self) -> dict[int, Tensor]:
        id_to_label = {}
        for anno in self.annos:
            for _id, label in zip(anno["ids"], anno["labels"]):
                if int(_id) not in id_to_label:
                    id_to_label[int(_id)] = [int(label)]
                else:
                    id_to_label[int(_id)].append(int(label))

        for _id, labels in id_to_label.items():
            id_to_label[_id] = torch.tensor(labels).cpu().long().mode().values # pervent class label switching

        return id_to_label
    
    def get_annotations(self, index: int = None):
        """
        concatenate annotations to a tensor with a shape of [N_obj, 7]. Concatenation order of each object:
        [label, id, box, area] (1 + 1 + 4 + 1)
        Args:
        ---
            - index: specifies objects from frame index. in not given, return all objects from the video clip.
        """
        keys = ["labels", "ids", "boxes", "area"]
        if index:
            return torch.cat([self.annos[index][key].float() for key in keys], dim=1)
        
        return torch.cat(
                [
                    torch.cat([self.annos[i][key].float() for key in keys],dim=1)
                    for i in range(self.num_images)
                ], dim=0
            )
    
    def get_labels(self, ids: Union[int, Iterable] = None) -> Tensor:
        """
        Get class labels of specified ids

        Args:
        ---
            - ids: object id. if None, returns labels of all instances
        """
        if not ids:
            return torch.tensor(self.id_to_index.values()).long()
        
        if not isinstance(ids, Iterable):
                ids = [ids]

        labels = [self.id_to_index[_id] for _id in ids]

        return torch.tensor(labels).flatten().long().to(self.appearence.device)
    
    def get_appearence(self, iter_index: int, ids: Union[int, Iterable] = None) -> Tensor:
        """
        Get appearence of specified object(s) in a clip. i.e. whether instances appear in a video or sliced video.

        Args:
        ---
            - iter_index: specifies if instances appear in which iteration / video slice. i.e.
                    `video[iter_index: num_frames + iter_index]`
            - ids: instance ids, which specifies which instances apparence info should be returned. 
                If not given, returns apparence info of all instances.

        Returns:
        ---
            - a boolean tensor with shape of [N_objs, 1] in which element with value of `True` denotes 
                appearing, `False` for disappearing.
        """
        assert iter_index < self.num_iters, "iter index out of range: [0, {}]".format(self.num_iters - 1)
        if ids is None:
            _app = self.appearence
        else:
            if not isinstance(ids, Iterable):
                ids = torch.tensor([ids]).long().to(self.appearence.device)

            indices = [self.id_to_index[_id.item()] for _id in ids]
            _app = self.appearence[indices]

        if iter_index is None:
            duration_ap = _app
        else:
            duration_ap = _app[:, iter_index: self.num_frames + iter_index]

        duration_ap = duration_ap.float().sum(-1, keepdim=True) > 0.

        return duration_ap.bool()

    
    def to(self, device: str) -> "GroundTruth":
        if isinstance(self.images, list):
            self.images = [img.to(device) for img in self.images]
        else:
            self.images = self.images.to(device)

        self.appearence = self.appearence.to(device)

        for key, val in self.id_to_label.items():
            self.id_to_label[key] = val.to(device)

        for anno in self.annos:
            for key, value in anno.items():
                if isinstance(value, Tensor):
                    anno[key] = value.to(device)

        return self

    def cpu(self):
        return self.to("cpu")

    def cuda(self):
        return self.to("cuda")

    def mps(self):
        return self.to("mps")
    

class BatchedOutputs:
    """
    A `BatchedOutputs` defines a model's output sturcture / interface.
    """
    training: bool = False
    valid_keys = ["pred_logits", "pred_boxes", "pred_objs", "hs", "query_spatial_pos"]
    optional_keys = ["aux_outputs"]
    def __init__(self, outs: dict[str, Tensor], num_detect_queries: int, iter_index: int = None) -> None:
        """
        Args:
        ---
            - outs: a dictionary output by a model which must contain keys:
                - `pred_logits`: unnormalized scores, shape [B, N, N_cls]
                - `pred_boxes`: boxes with a format of [cx, cy, w, h], shape [B, N, 4]
                - `pred_objs`: unnormalized objectness scores, shape [B, N, 1]
                - `hs`: hidden states, shape [B, N, D]
                - `query_spatial_pos`: spatial position embedding for query, shape [B, N, D]

            where `N` in dimension denotes sum of number of detect querys and track querys.

            - num_detect_queries: number of detect queries defined in the model, this is used as 
                an index for splitting track querys and detect querys.
            - iter_index: index of iteration of a video clip.
            
        both dictionaries must contain keys: `pred_logits`, `pred_boxes`, `pred_objs`, `hs`.
        """
        self.outs = outs
        self.num_detect_queries = num_detect_queries
        self.iter_index = iter_index if iter_index else 0

        self.batch_size, self.num_queries = self.outs[self.valid_keys[0]].shape[:2]
        self.has_track_queries = self.num_queries > num_detect_queries
        self.num_track_queries = self.num_queries - num_detect_queries
        self.has_aux_outputs = self.optional_keys[0] in self.outs
        self.aux_outputs: list[BatchedOutputs] = self.outs[self.optional_keys[0]] if self.has_aux_outputs else None

        self.detection_logits, self.detection_boxes, self.detection_objs, self.detection_hs, self.detection_pos_embed = [
            self.outs[key][:, :self.num_detect_queries] for key in self.valid_keys
        ] # [B, num_detect_queries, ...]

        self.num_classes = self.detection_logits.size(-1) - 1

        self.track_logits, self.track_boxes, self.track_objs, self.track_hs, self.track_pos_embed = [
            self.outs[key][:, self.num_detect_queries:] for key in self.valid_keys
        ] if self.has_track_queries else [
            torch.empty([0, i]).float().to(self.detection_boxes.device)
            for i in [self.detection_logits.size(-1), 4, 1, self.detection_hs.size(-1), self.detection_hs.size(-1)]
        ]

        self.query_spatial_pos = self.outs[self.valid_keys[-1]]

        # FIXME: redundant storage of indices
        self.track_padding_mask: Tensor = None
        self.track_gt_maps: list[tuple[Tensor, Tensor, Tensor, Tensor]] = None
        self.detect_gt_maps: list[tuple[Tensor, Tensor, Tensor]] = None
        self.track_ids: Tensor = None

    def __len__(self) -> int:
        return self.batch_size
    
    def __repr__(self) -> str:
        return "{}(num_queries={}, has_track_queries={}, hidden_dim={}, has_aux_output={})".format(
            self.__class__.__name__,
            self.num_queries, self.has_track_queries, self.detection_hs.size(-1), self.optional_keys[0] in self.outs
        )

    def set_iter_index(self, __index: int, /):
        self.iter_index = __index

        if self.has_aux_outputs:
            for out in self.aux_outputs:
                out.set_iter_index(__index)

    def set_track_padding_mask(self, __mask: Tensor, /):
        self.track_padding_mask = __mask

        if self.has_aux_outputs:
            for out in self.aux_outputs:
                out.set_track_padding_mask(__mask)

    def set_track_gt_maps(self, __map: list[tuple[Tensor, Tensor, Tensor, Tensor]] , /):
        assert len(__map) == self.batch_size
        self.track_gt_maps = __map

    def set_track_ids(self, __ids: Tensor, /):
        self.track_ids = __ids

    def set_detect_gt_maps(self, __map: list[tuple[Tensor, Tensor]], /):
        assert len(__map) == self.batch_size
        self.detect_gt_maps = __map

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def inference(self):
        self.training = False