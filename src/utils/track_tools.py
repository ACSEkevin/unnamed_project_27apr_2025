import torch

from torch import Tensor
from torch.nn import functional as F
from collections import deque
from typing import overload, Union

from .api import BatchedOutputs


class TrackStatus:
    NEWBORN = 0
    MATURE = 1
    DEAD = 2


class HistoryDeque:
    def __init__(self, max_len=10, device: str = None):
        self._que: deque[dict] = deque(maxlen=max_len)
        self.device = device if device else "cpu"

    def __len__(self) -> int:
        return len(self._que)
    
    def __getitem__(self, index: int) -> dict:
        return self._que[index]
    
    def to(self, device: str) -> "HistoryDeque":
        self.device = device

        for info in self._que:
            for key, val in info.items():
                if isinstance(val, Tensor):
                    info[key] = val.to(self.device)

        return self

    def push(self, box: Tensor, score: float, query: Tensor, objness: bool, durational_occ: bool) -> None:
        self._que.append(
            dict(box=box.to(self.device), score=score, 
                 query=query.to(self.device), objness=objness, durational_occ=durational_occ)
        )


class Track(object):
    """
    Track Structure for storing a single track instance information.
    """
    def __init__(self, id: int, class_id: int, box: Tensor, score: float, query: Tensor, pos_embed: Tensor, max_len: int = 10, device: str = None) -> None:
        """
        Args:
        ---
            - id: track id
            - class_id: class/category index
            - box: bounding box with format `(cx, cy, w, h)`
            - score: predict confidence score by a detector/tracker
            - query: track query / embedding, shape: [D]
            - pos_embed: positional embedding of the track, [D]
            - max_len: max number of history info storage
        """
        self._id = id
        self._class_id = class_id
        self._box = box
        self._score = score
        self._query = query
        self._pos_embed = pos_embed

        self._objness: bool = True # instantiated Track must exist
        self._durational_occ: bool = True

        self._age: int = 1
        self._disappear_times: int = 0

        self.device = device if device else "cpu"
        self._history = HistoryDeque(max_len=max_len).to(self.device)

    def __repr__(self) -> str:
        return "{}(id={}, cls={}, score={}, objness={})".format(
            self.__class__.__name__,
            self._id, self._class_id, self._score, self._objness
        )
    
    def to(self, device: str) -> "Track":
        self._history = self._history.to(device)
        self._box = self._box.to(device)
        self._query = self._query.to(device)
        self._pos_embed = self._pos_embed.to(device)
        self.device = device

        return self
    
    def update(self, box: Tensor, score: float, query: Tensor, pos_embed: Tensor, objness: bool, durational_occ: bool):
        self._box = box
        self._score = score
        self._query = query
        self._pos_embed = pos_embed
        self._objness = objness
        self._durational_occ = durational_occ

        if not self._objness and not self._durational_occ:
            self._disappear_times += 1
        else:
            self._age += 1
            self._disappear_times = 0

        self._history.push(box, score, query, objness, durational_occ)

    def is_appeared(self, history_idx: int = None) -> bool:
        """
        Is appeared in current frame or specified history frame.
        """
        if not history_idx:
            history_idx = -1

        return self._history[history_idx]["objness"]

    def get_state(self, index: int) -> dict:
        return self._history[index]

    def reset(self):
        self._age = 0
        self._disappear_times: int = 0

    @property
    def id_(self):
        return self._id
    
    @property
    def class_id_(self):
        return self._class_id
    
    @property
    def box_(self):
        return self._box
    
    @property
    def score_(self):
        return self._score
    
    @property
    def query_(self):
        return self._query
    
    @property
    def pos_embed_(self):
        return self._pos_embed

    @property
    def age_(self):
        return self._age
    
    @property
    def disappear_times_(self):
        return self._disappear_times
    
    @property
    def durational_occ_(self):
        return self._durational_occ


class TrackManager:
    """
    A class managing a number of track instances from one particular image sequence.
    """
    def __init__(self, max_disappear_times: int, track_max_history: int = 10, device: str = None) -> None:
        self.max_disappear_times = max_disappear_times
        self.track_max_history = track_max_history

        self._id_pool: deque = deque([i for i in range(9999)])
        self._track_pool: dict[int, Track] = {}
        self.device = device if device else "cpu"

    def __len__(self) -> int:
        return len(self._track_pool)
    
    def __getitem__(self, track_id: int) -> Track:
        return self._track_pool[track_id]
    
    def to(self, device: str) -> "TrackManager":
        self._track_pool = {
            track_id: track.to(device)
            for track_id, track in self._track_pool.items()
        }
        self.device = device

        return self
    
    def reset(self):
        self._id_pool = deque([i for i in range(9999)])
        self._track_pool.clear()
    
    def get_track(self, track_id: int) -> Track:
        return self._track_pool[track_id]
    
    def update_track_state(self, track_id: int, box: Tensor, score: float, 
                           query: Tensor, pos_embed: Tensor, objness: bool, durational_occ: bool) -> None:
        track = self.get_track(track_id)
        track.update(box, score, query, pos_embed, objness, durational_occ)

    def create_new_track(self, class_id: int, box: Tensor, score: float, query: Tensor, pos_embed: Tensor) -> None:
        track_id = self._id_pool.popleft()
        assert track_id not in self._track_pool, "Track {} still exists in pool.".format(track_id)

        track = Track(track_id, class_id, box, score, query, pos_embed, self.track_max_history, device=self.device)
        self._track_pool[track_id] = track

    def delete_disappeared_tracks(self) -> int:
        num_deleted_tracks = 0
        for track_id, track in self._track_pool.items():
            if track.disappear_times_ >= self.max_disappear_times:
                self._id_pool.append(track_id) # recycle track ids
                self._track_pool.pop(track_id)
                num_deleted_tracks += 1

        return num_deleted_tracks
    
    def create(self, info: Tensor, queries: Tensor, pos_embeds: Tensor):
        """
        Create new tracks based on given infos.

        Args:
        ---
            - info: [N, 6] with dtype of `float`, `6` in dim 2 represents [class_id, score, box].
            - queries: shape [N, D]
            - pos_embeds: shape [N, D]
        """
        assert info.size(0) == queries.size(0) == pos_embeds.size(0), "{} vs. {} vs. {}".format(
            info.size(0), queries.size(0), pos_embeds.size(0)
        )

        dim = queries.size(-1)
        split_size = [1, 1, 4, dim, dim]
        all_info = torch.cat([info, queries, pos_embeds], dim=-1) # [N, 6 + D + D]

        for cls_id, score, box, query, pos_embed in zip(*all_info.split(split_size, dim=-1)):
            self.create_new_track(
                cls_id.long().item(), box, score.item(), query, pos_embed
            )
    
    def update(self, info: Tensor, queries: Tensor, pos_embeds: Tensor):
        """
        Update existing track infos.

        Args:
        ---
            - info: [N, 8] with dtype of `float`, `8` in dim 2 represents 
                [track_id, score, box, objness, durational_occ].
            - queries: shape [N, D]
            - pos_embeds: shape [N, D]
        """
        assert info.size(0) == queries.size(0) == pos_embeds.size(0), "{} vs. {} vs. {}".format(
            info.size(0), queries.size(0), pos_embeds.size(0)
        )
        dim = queries.size(-1)
        split_size = [1, 1, 4, 1, 1, dim, dim]
        all_info = torch.cat([info, queries, pos_embeds], dim=-1) # [N, 8 + D + D]

        for track_id, score, box, objness, d_occ, query, pos_mebed in zip(*all_info.split(split_size, dim=-1)):
            self.update_track_state(
                track_id.long().item(), box, score.item(), query, pos_mebed, 
                objness.bool().item(), d_occ.bool().item()
            )

    def pack_boxes(self):
        """
        Pack all track boxes together as a `torch.Tensor`.

        Returns:
        ---
            - boxes, shape: [N, 4]
        """
        if self.is_empty():
            return torch.empty([0, 4], device=self.device).float()
        
        return torch.stack([track.box_  for _, track in self._track_pool.items()], dim=0) # [N, 4]

    def pack_queries(self):
        """
        Pack all tracks and OUTPUT track queries concatenated with position embeddings, and corresponding ids.

        Returns:
        ---
            - track_with_pos_embed, shape [N, 512]
            - track_ids
        """
        if self.is_empty():
            track_with_pos_embed = torch.empty([0, 512], device=self.device).float()
        else:
            track_with_pos_embed = torch.stack([
                torch.cat([track.query_, track.pos_embed_], dim=0) for _, track in self._track_pool.items()
            ], dim=0) # [N, 512]

        return track_with_pos_embed, torch.tensor(list(self._track_pool.keys())).long().to(self.device)
    
    def is_empty(self):
        return len(self) == 0
    
    @property
    def existing_track_ids_(self):
        return list(self._track_pool.keys())
