from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ConstantLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import torch, os, shutil

from src.utils.api import BatchedOutputs, GroundTruth
from src.utils.misc import NestedTensor
from src.utils.track_tools import TrackManager
from src.utils.box_ops import box_iou, box_cxcywh_to_xyxy
from src.models import UnnamedModel
from src.losses import Matcher, CollectiveLoss
from src.metrics import MOTMetricWrapper


# TODO: auto mixed precision training at cuda divice
class Trainer:
    """
    Define the procedure of training process of the model.
    """
    def __init__(self,
                 model: UnnamedModel,
                 loss: CollectiveLoss,
                 optim: Optimizer,
                 lr_scheduler: LRScheduler = None,
                 max_norm: float = 0.,
                 sample_fp_prob: float = 0.,
                 sample_max_score: float = 1.,
                 drop_rate: float = 0.,
                 max_disappear_times: int = 4,
                 max_track_history: int = 10,
                 cls_conf: float = 0.5,
                 dur_occ_conf: float = 0.5,
                 matching_thres: float = None,
                 log_dir: str = None
                 ) -> None:
        
        self.model = model
        self.loss_fn = loss
        self.optim = optim
        self.lr_scheduler = lr_scheduler if lr_scheduler else ConstantLR(self.optim, factor=1., total_iters=1)
        self.max_norm = max_norm
        self.sample_fp_prob = sample_fp_prob
        self.sample_max_score = sample_max_score
        self.drop_rate = drop_rate
        self.log_dir = log_dir

        self.matcher = Matcher(
            loss.cls_loss_weight, 
            loss.box_l1_loss_weight, 
            loss.box_giou_loss_weight
        )

        self.num_frames = model.num_frames
        self.epoch_start = 0
        self.current_epoch = 0
        self.device: str = "cpu"

        self.evaluator = Evaluator(
            self.model, max_disappear_times, max_track_history, 
            cls_conf, dur_occ_conf, matching_thres, 
            device=self.device
        )

        self._writer: SummaryWriter
        if self.log_dir:
            if os.path.exists(self.log_dir): # reset dir
                shutil.rmtree(self.log_dir)
            os.mkdir(self.log_dir)
            self._writer = SummaryWriter(self.log_dir)

    @staticmethod
    def _sample_false_positive_tracks(pred: BatchedOutputs, matched_detect_results: list[tuple[Tensor, Tensor, Tensor]], prob: float, max_score: float = 1.) -> list[Tensor]:
        """
        Sample false positive queries from detect queries with a specified probability

        Args:
        ---
            - pred: a `BatchedOutputs` structure from model output, with matching indices set.
            - matched_detect_result: matched detections (output indices, gt indices, gt ids) from matcher
            - prob: probability of selection of each query.
            - max_score: the selected inactive queries' max class score, tracks with scores higher
                than which will not be selected. This is for hard-negative samples mining.

        Returns:
        ---
            - a list(batch) of indices meaning queries will be selected.
        """
        assert 0. <= prob <= 1.
        assert 0. <= max_score <= 1.

        sampled_indices = []

        for batch_idx in range(pred.batch_size): # batch-wise sample
            matched_hs_indices = matched_detect_results[batch_idx][0]
            indices = torch.arange(pred.num_detect_queries).to(pred.detection_boxes.device).long() # indices

            # FIXME: logistic output
            scores = pred.detection_logits[batch_idx].sigmoid().max(-1).values # [N, N_cls] -> [N]
            indices_higher_score = torch.where(scores >= max_score)[0] # queries with scores over max_score
            invalid_indices = torch.cat([matched_hs_indices, indices_higher_score], dim=0) # [N_invalid]
            indices[invalid_indices] = -1 # invalid indices (over scored & matched) are marked as negative

            valid_indices = indices[indices >= 0] # [N - N_invalid]
            sampled_indices.append(
                valid_indices[valid_indices.bernoulli(p=prob).bool()].long() # select queries as fp tracks by prob
            )

        return sampled_indices
    
    @staticmethod
    def _random_drop_active_tracks(matched_track_indices: Tensor, matched_track_ids: Tensor, drop_rate: float):
        """
        Drop active track queries with a specified probability. This is a trick of 
        track query augmentation, to do such that tracks could be generated from 
        new detect queries.

        Args:
        ---
            - matched_track_indices: matched indices of track queries
            - matched_track_ids: matched tracks' corresponding ground truth ids.
            - drop_rate: rate / probability of track dropping
        """
        indices = torch.arange(matched_track_indices.numel()).long().to(matched_track_indices.device)
        keep = indices.bernoulli(1 - drop_rate + 1e-8).bool()

        return matched_track_indices[keep], matched_track_ids[keep]
    
    @torch.no_grad()
    def _enter_and_exit_queries(self,
                            pred: BatchedOutputs,
                            targets: list[GroundTruth],
                            matched_detect_results: list[tuple[Tensor, Tensor, Tensor]] = None,
                            matched_track_results: list[tuple[Tensor, Tensor, Tensor, Tensor]] = None,
                            fp_prob: float = 0., 
                            fp_max_score: float = 1.,
                            drop_prob: float = 0.
                            ):
        """
        Build new track queries (enter) from new detections and delete those diappeared within a period (exit).
        This process contains track query augmentations: sampling false positive tracks and random dropping tracks.

        Args:
        ---
            - pred: a `BatchedOutputs` structure from model output
            - targets: a list of `GroundTruth` structure
            - matched_detect_results: matched detections (output indices, gt indices, gt ids) from matcher
            - matched_track_results: matched tracks (track query indices, gt indices, gt ids) from matcher
            - fp_prob: probability of sampling false positive tracks
            - fp_max_score: the selected inactive queries' max class score, tracks with 
                scores higher than which will not be selected.
            - drop_prob: probability of track dropping.

        Returns:
        ---
            - a `NestedTensor` containing variables:
                - `tensors`: batched padded track queries concatenated with position embedding, shape [B, N_, 2D]
                - `mask`: a mask in which elements with value of `0` indicate active tracks, `-1`: false positives
                    and `-2`: padded tracks, shape: [B, N_]
            - updated `matched_track_results`
        """
        device = pred.detection_boxes.device
        # first iteration, track queries have not generated
        # 1. calculate number of track queries in this time stamp
        num_matched_indices = torch.tensor(
            [res[0].numel() for res in matched_detect_results], device=device
        ).long() # matched detections / new tracks in current tme stamp, [B]

        if pred.has_track_queries: # if has track queries inherited from last time stamp
            num_matched_track_indices = torch.zeros(pred.batch_size).long().to(device)
            for index in range(pred.batch_size):
                track_indices, gt_indices, track_ids, keep = matched_track_results[index]
                # get track occurence in a period (video clip), then drop disappeared ones
                valid_mask = targets[index].get_appearence(pred.iter_index, track_ids).flatten()
                updated_track_indices, updated_track_ids = track_indices[valid_mask], track_ids[valid_mask] # exit instances
                # randomly drop tracks
                if drop_prob > 0.:
                    updated_track_indices, updated_track_ids = self._random_drop_active_tracks(
                        updated_track_indices, updated_track_ids, drop_prob
                    )

                num_matched_track_indices[index] = updated_track_indices.numel()
                matched_track_results[index] = (updated_track_indices, gt_indices, updated_track_ids, keep) # cover track indices and ids

            num_matched_indices += num_matched_track_indices # plus inherited existing tracks, [B]

        if fp_prob > 0.: # if samples fp tracks
            selected_indices = self._sample_false_positive_tracks(pred, matched_detect_results, fp_prob, fp_max_score)
            num_selected_indices = torch.tensor(
                [indices.numel() for indices in selected_indices], device=device
            )

            num_matched_indices += num_selected_indices # plus number of fp tracks, [B]

        # so far, num_matched_indices = num_detect_queries + num_inherited_tracks (if exists) + fp_tracks (if sampled)
        # 2. build a padded tensors and mask with dimension defined
        padding_length = num_matched_indices.max() # get maximum length which indicates size in tensor dim 1
        tensors = torch.zeros(
            [pred.batch_size, int(padding_length), pred.detection_hs.size(-1) * 2], device=device
        ).float() # [B, N_t + N_fp + N_pad, D]
        mask = torch.full(
            [pred.batch_size, int(padding_length)], -1., device=device).float() # init as -1 [B, N_t + N_fp + N_pad]

        # 3. fill-in tracks to tensors, mask
        updated_matched_track_results: list[tuple[Tensor, Tensor, Tensor, Tensor]] = []
        for index in range(pred.batch_size):
            # new detections to tracks
            detect_matched_indices, _, detect_matched_ids = matched_detect_results[index]
            active_queries = pred.detection_hs[index, detect_matched_indices] # [N_d, D]
            active_pos_embed = pred.detection_pos_embed[index, detect_matched_indices]
            updated_track_ids = detect_matched_ids
            updated_track_indices = torch.arange(detect_matched_indices.numel()).long().to(device)

            # exiting tracks
            if pred.has_track_queries:
                updated_track_indices, _, updated_track_ids, _ = matched_track_results[index] # remaining track indices
                # fp_padding_mask = pred.track_padding_mask[index] == 0 # [N_t + N_fp + N_pad]

                inherited_track_queries = pred.track_hs[index, updated_track_indices]
                inherited_pos_embed = pred.track_pos_embed[index, updated_track_indices]

                # update indices and ids
                updated_track_ids = torch.cat([updated_track_ids, detect_matched_ids])
                updated_track_indices = torch.arange(updated_track_ids.numel()).long().to(device) # new indices
                # add new queries and pos_embed
                active_queries = torch.cat([inherited_track_queries, active_queries]) # [N_e_ + N_d, D]
                active_pos_embed = torch.cat([inherited_pos_embed, active_pos_embed])

            updated_matched_track_results.append(
                # [N_e_ + N_d] or [N_d], matched gt indices - do not care, [N_e_ + N_d] or [N_d], valid mask - do not care
                (updated_track_indices, torch.empty(0).long().to(device), updated_track_ids, torch.empty(0).bool().to(device))
            )
                    
            mask[index, :active_queries.size(0)] = 0. # active ones are marked as `0`

            if fp_prob > 0: # if sampled fp tracks
                fp_indices = selected_indices[index]
                fp_queries = pred.detection_hs[index, fp_indices] # [N_fp, D]
                fp_pos_embed = pred.query_spatial_pos[index, fp_indices]
                active_queries = torch.cat([active_queries, fp_queries]) # [N_e_ + N_d + N_fp, D]
                active_pos_embed = torch.cat([active_pos_embed, fp_pos_embed]) # [N_e_ + N_d + N_fp, D]

            tensors[index, :num_matched_indices[index]] = torch.cat([active_queries, active_pos_embed], dim=1) #  # [N_e_ + N_d + N_fp, 2D]
            mask[index, active_queries.size(0):] = -2. # padded are marked as -2, rest are fp tracks (-1).

        return NestedTensor(tensors, mask), updated_matched_track_results
    
    def reset(self):
        self.epoch_start = 0
        self.current_epoch = 0

        self.model.reset_prefill()

        if self.log_dir:
            self._writer.close()
    
    def to(self, device: str) -> "Trainer":
        self.model.to(device)
        self.loss_fn.to(device)
        self.evaluator.to(device)
        self.device = device

        return self
    
    def save_checkpoint(self, path: str):
        state_dict = dict(
            model=self.model.state_dict(),
            optimizer=self.optim.state_dict(),
            lr_scheduler=self.lr_scheduler.state_dict(),
            epoch=self.current_epoch
        )
        torch.save(state_dict, path)

    def load_checkpoint(self, path: str, weights_only: bool = True):
        ckpt = torch.load(path, map_location="cpu", weights_only=weights_only)
        self.model.load_state_dict(ckpt["model"])
        self.optim.load_state_dict(ckpt["optimizer"])
        self.lr_scheduler.load_state_dict(ckpt["lr_scheduler"])

        self.epoch_start = ckpt["epoch"]
        self.current_epoch = ckpt["epoch"]

    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader = None, num_epochs: int = 1, 
              eval_freq: int = 0, save_freq: int = 0, save_dir: str = None, device: str = None):
        """
        Execute model training.

        Args:
        ---
            - train_dataloader: a `DataLoader` for training
            - val_dataloader: a `DataLoader` with loaded validation dataset, turning on evaluation depends on arg `eval_freq`
            - num_epochs: number of epochs
            - eval_freq: specify evaluating model after how many training epochs, if set `eval_freq=0` implies does not evaluate
            - save_freq: After how many training epochs saves checkpoints, this includes saving states of model, optimizer, 
                scheduler and epoch. `save_freq=0` means does not save
            - save_dir: checkpoint save path. If `save_freq=0`, this term will be ignored. If save is checkpoint set to `on`
                and this arg is not given, then the trainer will create a path to save
            - device: training device, also can be set by `Trainer.to()`
        """
        self.on_train_begin()

        if device is not None:
            self.device = device

        if not self.epoch_start == 0:
            print("model will trained from epoch {}".format(self.epoch_start + 1))

        for epoch in range(self.epoch_start, num_epochs):
            print("\nepoch {}/{}".format(epoch + 1, num_epochs))

            self.train_one_epoch(train_dataloader)

            if val_dataloader is not None and eval_freq > 0 and (epoch + 1) % eval_freq == 0:
                eval_result = self.evaluator.evaluate(val_dataloader, report_result=True)

                if self.log_dir:
                    for name, value in zip(eval_result.columns, eval_result.values[0]):
                        self._writer.add_scalar(
                            f"eval/{name}", 
                            value, 
                            self.current_epoch + 1
                        )

            if save_freq > 0 and (epoch + 1) % save_freq == 0:
                if save_dir is None:
                    save_dir = os.path.join(os.getcwd(), "checkpoints")
                    if not os.path.exists(save_dir):
                        os.mkdir(save_dir)
                    save_path = os.path.join(save_dir, f"{self.model.name}_epoch_{epoch + 1}.pth")

                self.save_checkpoint(save_path)

        self.on_train_end()

    def train_one_epoch(self, dataloader: DataLoader):
        self.on_epoch_begin()
        pbar = tqdm(dataloader, total=len(dataloader), ncols=90)

        for index, targets in enumerate(pbar):
            pbar.set_description("train")
            targets: list[GroundTruth] # typedef
            postfix_kwargs = {}

            loss_mat = self.train_one_step(targets)

            for loss_name, value in zip(["cls_loss", "obj_loss", "reg_loss"], loss_mat):
                postfix_kwargs.update({loss_name: value.item()})

                if self.log_dir:
                    self._writer.add_scalar(
                        f"train/{loss_name}", 
                        value.item(), 
                        self.current_epoch * len(dataloader) + index + 1
                    )

            pbar.set_postfix(postfix_kwargs)

        self.on_epoch_end()

    def train_one_step(self, inputs: list[GroundTruth]):
        self.on_step_begin()

        assert inputs[0].num_frames == self.model.num_frames
        inputs = [i.to(self.device) for i in inputs]
        
        images = torch.stack([gt.images for gt in inputs], dim=0) # [B, T, 3, H, W]
        prev_track_matching_res: list[tuple[Tensor, Tensor, Tensor, Tensor]] = None
        track_queries_with_pos: NestedTensor = None

        for iter_index in range(inputs[0].num_iters):
            if iter_index == 0:
                batched_inputs = images[:, :self.num_frames]  # pre-fill, [B, num_frames, 3, H, W]
            else:
                # single-image inference [B, 1, 3, H, W]
                batched_inputs = images[:, self.num_frames + iter_index - 1: self.num_frames + iter_index]

            preds = self.model.forward(batched_inputs, track_queries_with_pos)
            preds.set_iter_index(iter_index)

            if iter_index > 0: # might contains track queries
                preds.set_track_gt_maps(prev_track_matching_res) # set last matched ground truth ids

            detect_res, track_res = self.matcher.forward(preds, inputs) # matcher only matches detect queries
            self.loss_fn.update(preds, inputs, detect_res, track_res) # caculate loss

            track_queries_with_pos, prev_track_matching_res = self._enter_and_exit_queries(
                preds, inputs, detect_res, track_res, 
                self.sample_fp_prob, self.sample_max_score, self.drop_rate
            ) # sample fp & padding

        loss_mat = self.loss_fn.unsummed_result() # returned for loss diaplaying
        loss = loss_mat.sum() # cls + obj + reg loss
        loss.backward()

        if self.max_norm > 0: # clip gradient norm
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)

        self.optim.step()
        self.on_step_end()

        return loss_mat
    
    def on_train_begin(self):...
    def on_train_end(self):...

    def on_epoch_begin(self):
        self.model.train()
        self.loss_fn.train()

    def on_epoch_end(self):
        self.current_epoch += 1
        self.lr_scheduler.step()
        self.model.reset_prefill()

    def on_step_begin(self):
        self.model.reset_prefill()
        self.optim.zero_grad()

    def on_step_end(self):
        self.loss_fn.reset()


class Evaluator:
    """
    Evaluate the model.\\
    NOTE that evalutation batch must set to 1, i.e. evaluate on
    a single video / video clip.
    """
    def __init__(self,
                 model: UnnamedModel,
                 max_disappear_times: int,
                 max_track_history: int = 10,
                 cls_conf: float = 0.5, 
                 dur_occ_conf: float = 0.5,
                 matching_thres: float = None,
                 device: str = None
                 ) -> None:
        
        self.model = model
        self.max_disappear_times = max_disappear_times
        self.max_track_history = max_track_history
        self.cls_conf = cls_conf
        self.dur_occ_conf = dur_occ_conf
        self.matching_thres = matching_thres
        self.device = device if device else "cpu"

        self.num_frames = model.num_frames
        self.met = MOTMetricWrapper(matching_thres=matching_thres)
        self.track_manager = TrackManager(max_disappear_times, max_track_history)

    def _calculate_distance_mat(self, iter_index: int, target: GroundTruth, dist_thres: float = None):
        """
        Calculate distance matrix for evaluating.
        """

        track_boxes = self.track_manager.pack_boxes() # [N_t, 4]
        annos = target.get_annotations(target.num_frames + iter_index - 1) # [N_gt, ...]
        gt_ids, gt_boxes = annos[:, 1], annos[:, 2:6]

        iou, _ = box_iou(
            box_cxcywh_to_xyxy(gt_boxes),
            box_cxcywh_to_xyxy(track_boxes)
        )

        if dist_thres:
            iou[iou >= dist_thres] = torch.nan # [N_gt, N_t]

        return gt_ids, iou

    def _select_active_queries(self, out: BatchedOutputs, cls_conf: float, dur_occ_conf: float):
        """
        Select active queries by given class and durational occurence confidence scores.

        Args:
        ---
            - out: a `BatchedOutputs` structure with track ids set.
            - cls_conf: class confidence threshold, instance score lower than which 
                is considered 'inactive' in current frame
            - dur_occ_conf: occurence in a duration confidence threshold, instance score
                lower than which is considered 'disappeared' in a video clip

        Returns:
        ---
            - output from `.track_tools.TrackManager.pack()`
            - number of deleted tracks
            - number of new tracks
        """

        #1. update status of existing tracks
        if out.has_track_queries: # has tracks from last frame
            assert out.track_ids is not None # [N_t]
            # FIXME: logistic output
            track_cls_scores = out.track_logits.squeeze().sigmoid().max(-1).values # [N_t]
            cur_objness = track_cls_scores > cls_conf # [N_t]
            d_occ = out.track_objs[0].sigmoid() > dur_occ_conf # [N_t, 1]

            self.track_manager.update(
                torch.cat([ # FIXME: might occur size shrinking [N1, N2] -> [N1, ]
                    out.track_ids.float().unsqueeze(-1),
                    track_cls_scores.unsqueeze(-1),
                    out.track_boxes.squeeze(),
                    cur_objness.float().unsqueeze(-1),
                    d_occ.float(),
                ], dim=1),
                out.track_hs.squeeze(),
                out.track_pos_embed.squeeze()
            )

        # 2. deleted disappeared tracks base on track info from step 1
        num_deleted_tracks = self.track_manager.delete_disappeared_tracks()

        # 3. create new tracks from new detections
        # FIXME: logistic output
        cls_scores, cls_indices = out.detection_logits.squeeze().sigmoid().max(-1) # [1, N_d, N_cls + 1] -> [N_d, N_cls] -> [N_d]
        keep = cls_scores > cls_conf

        filtered_cls_indices, filtered_scores = cls_indices[keep], cls_scores[keep] # [N_d], [N_d]
        filtered_boxes = out.detection_boxes.squeeze()[keep] # [N_d, 4]
        filered_hs, filtered_pos = out.detection_hs.squeeze()[keep], out.detection_pos_embed.squeeze()[keep] # [N_d, 256]

        # TODO: add to log
        # max_bg = out.detection_logits.squeeze().softmax(-1)[..., -1].median()
        # print("min scores: {:.3f}, max_scores: {:.3f}, bg median: {:.3f}".format(cls_scores.min().item(), cls_scores.max().item(), max_bg.item()))

        self.track_manager.create(
            torch.cat([
                filtered_cls_indices.float().unsqueeze(-1),
                filtered_scores.unsqueeze(-1),
                filtered_boxes,
            ], dim=1),
            filered_hs, filtered_pos
        )
        track_with_pos_embed, track_ids = self.track_manager.pack_queries() # [N, 512], ...

        return NestedTensor(track_with_pos_embed.unsqueeze(0), None), track_ids, num_deleted_tracks, filtered_cls_indices.numel()

    def to(self, device: str) -> "Evaluator":
        self.model.to(device)
        self.track_manager = self.track_manager.to(device)
        self.device = device

        return self

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader, report_result: bool = False):
        """
        Perform a batched evaluation

        Args:
        ---
            - dataloader: a `DataLoader` with loaded validation dataset, batch size set to 1.
            - report_result: whether print result to the terminal.
        """
        self.on_eval_begin()

        pbar = tqdm(dataloader, total=len(dataloader), ncols=90)

        for targets in pbar: # traverse one clip
            pbar.set_description("eval")
            targets: list[GroundTruth] # typedef
            assert len(targets) == 1, "evaluator requires batch size must be 1, but got {}".format(len(targets))
            assert targets[0].num_frames == self.model.num_frames

            self.evaluate_one_step(targets[0])

        result = self.met.result(std_out=report_result)
        self.on_eval_end()

        return result

    @torch.no_grad()
    def evaluate_one_step(self, target: GroundTruth):
        self.on_step_begin()

        target.to(self.device)
        target.images.unsqueeze_(0)  # [1, T, 3, H, W]
        track_queries_with_pos: NestedTensor = None
        track_ids: Tensor = None

        for iter_index in range(target.num_iters):
            if iter_index == 0:
                batched_inputs = target.images[:, :self.num_frames]  # pre-fill, [1, num_frames, 3, H, W]
            else:
                # single-image inference [1, 1, 3, H, W]
                batched_inputs = target.images[:, self.num_frames + iter_index - 1: self.num_frames + iter_index]

            preds = self.model.forward(batched_inputs, track_queries_with_pos)
            preds.set_iter_index(iter_index)
            if iter_index > 0:
                preds.set_track_ids(track_ids)

            track_queries_with_pos, track_ids, _, _ = self._select_active_queries(preds, self.cls_conf, self.dur_occ_conf)
            gt_ids, distance_mat = self._calculate_distance_mat(iter_index, target, self.matching_thres)

            self.met.update(gt_ids, track_ids, distance_mat)

        partial_result = self.met.partial_result(render=False)
        self.on_step_end()

        return partial_result

    def on_eval_begin(self):
        self.model.eval()

    def on_eval_end(self):
        self.met.reset()
        self.track_manager.reset()

    def on_step_begin(self):...
    def on_step_end(self):
        self.met.accumulate()
        self.track_manager.reset()
        self.model.reset_prefill()
    
