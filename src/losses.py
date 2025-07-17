"""

"""

from torch import nn, Tensor
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment
from typing import Callable

import torch

from .utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from .utils.api import BatchedOutputs, GroundTruth


class HungarianMatcher(nn.Module):
    """Copied and modified from DETR repo::[matcher.py](https://github.com/facebookresearch/detr/blob/main/models/matcher.py)
    
    This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object (ø). Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs: BatchedOutputs, targets: list[GroundTruth], invalid_target_indices: list[Tensor] = None):
        """ Performs the matching

        Params:
            - pred: a `BatchedOutput` data structure
            - target: a list of `GroundTruth`data structure
            - invalid_target_indices: specifies that targets which are not participated in matching.
                This is for targets aleardy associated with existing tracks.

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        device = outputs.detection_boxes.device
        bs, num_queries = outputs.batch_size, outputs.num_detect_queries
        current_index = outputs.iter_index + targets[0].num_frames - 1

        if invalid_target_indices is not None:
            assert len(invalid_target_indices) == len(targets)

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs.detection_logits.flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs.detection_boxes.flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes, annos [BM, 7] 7: [cls, ids, cx, cy, w, h, areas]
        annos = torch.cat([target.get_annotations(current_index) for target in targets])
        tgt_cls = annos[:, 0].long() # [BM]
        tgt_ids = annos[:, 1].long() # [BM]
        tgt_bbox = annos[:, 2:6] # [BM, 4]

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_cls] # range [-1, 0] shape [BN, BM]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1) # [BN, BM]

        # Compute the giou cost betwen boxes # [BN, BM]
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [target.num_objects[current_index] for target in targets] # number of objs in frames of each batch 
        idx_pairs: list[tuple[Tensor, Tensor, Tensor]] = []

        for index, (c, frame_target_ids)in enumerate(zip(C.split(sizes, -1), tgt_ids.split(sizes))): # c: [BN, ...]
            cost = c[index] # [N, M_index]
            if invalid_target_indices is not None:
                # set a high cost value to invalid indices
                # setting random values such that invalid columns are not always matched to front rows
                cost[:, invalid_target_indices[index]] = torch.randint(
                    2000, 3000, [cost.size(0), invalid_target_indices[index].numel()]
                ).float().cpu()

            hs_indices, gt_indices = [torch.as_tensor(idx).long() for idx in linear_sum_assignment(cost)]
            mask = cost[hs_indices, gt_indices] < 1000.
            hs_indices, gt_indices = [idx[mask].to(device) for idx in [hs_indices, gt_indices]] # drop invalid pairs.
            idx_pairs.append((hs_indices, gt_indices, frame_target_ids[gt_indices]))

        return idx_pairs
    

class Matcher(nn.Module):
    """
    Match all queries (track queries and detect queries) from a particular output.
    """
    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        super().__init__()

        self.detect_matcher = HungarianMatcher(cost_class, cost_bbox, cost_giou)
        self.prev_match_results = None

    @staticmethod
    def _get_tracks_matched_gt_indices(matched_ids: Tensor, gt_annos: Tensor):
        """
        Find existing tracks and corresponding ground truth indices.

        Args:
        ---
            - matched_ids: matched ground truths' ids of track queries. shape [N,]
            - gt_annos: annotations with a shape of [N_obj, 7]. where second dim: 
                [label, id, cx, cy, w, h, area]

        Returns:
        ---
            - indices: matched indices.
            - keep: a mask in which element with value of `False` denotes tracks
                disappears from matched ids, shape [N - N_obj]
        """
        device = matched_ids.device
        if gt_annos.numel() == 0:
            return torch.empty([0]).long().to(device), torch.zeros_like(matched_ids).bool().to(device)
        current_ids = gt_annos[:, 1].long()
        max_gt_id = current_ids.max().long().item()
        max_gt_id = max(max_gt_id, matched_ids.max().long()) # get current id range

        # id to index map
        id_map = torch.full([int(max_gt_id) + 1], -1).long().to(matched_ids.device)
        id_map[current_ids] = torch.arange(len(current_ids), device=matched_ids.device).long()
        
        # get matched indices, those disappeared are marked as -1
        indices = id_map[matched_ids]
        keep = (indices >= 0) & (indices < len(gt_annos))

        return indices[keep], keep
    
    @torch.no_grad()
    def forward(self, outputs: BatchedOutputs, targets: list[GroundTruth]):
        """
        Perform matching for new detections, Hungarian matching is applied, for those inherited 
        from last (tracks), check disappeared and update track and annotation indices.
        """
        # device = outputs.track_boxes.device
        current_index = outputs.iter_index + targets[0].num_frames - 1

        # if has track queries, split annotations to matched and unmatched.
        track_pairs: list[tuple[Tensor, Tensor, Tensor, Tensor]] = []
        invalid_gt_indices: list[Tensor] = [] 
        
        for batch_idx in range(outputs.batch_size):
            if outputs.has_track_queries: # has track queries, then split annotations
                hs_indices, _, gt_ids, _ = outputs.track_gt_maps[batch_idx]
                annos = targets[batch_idx].get_annotations(current_index)
                # find disappeared tracks
                existing_gt_indices, keep = self._get_tracks_matched_gt_indices(gt_ids, annos)

                # matched track results
                track_pairs.append((hs_indices, existing_gt_indices, gt_ids, keep))
                invalid_gt_indices.append(existing_gt_indices) # collect invalid indices for Hungarian matcher 
            else:
                invalid_gt_indices = None
                track_pairs = None

        detect_pairs = self.detect_matcher.forward(outputs, targets, invalid_gt_indices)
        
        return detect_pairs, track_pairs


class Loss(nn.Module):
    """
    Define forward interface of a loss function.
    """
    _identity: Callable[[Tensor], Tensor] = lambda x: x
    _reduce_map = dict(mean=torch.mean, sum=torch.sum, none=_identity)
    def __init__(self, reduction:str = "mean") -> None:
        super().__init__()
        assert reduction in self._reduce_map, "Unexpected reduction: {}".format(reduction)
        self.reduction = reduction

    def _reduce(self, loss: Tensor) -> Tensor:
        return self._reduce_map[self.reduction](loss)

    def forward(self, pred: Tensor, target: Tensor):
        raise NotImplementedError()
    

# FIXME: should generalize to logistic reg output (N_cls = 1) / multi-class classification
class ClassificationLoss(Loss):
    """
    Binary `focal` cross entropy loss. Copied & modified from [DETR repo](https://github.com/facebookresearch/detr/blob/main/models/segmentation.py)
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    """
    def __init__(self, focal: bool = False, alpha: float = 0.25, gamma: float = 2, reduction: str = "mean") -> None:
        """
        Args:
        ---
            - focal: whether uses focal loss, if `False`, args `alpha` and `gamma` will be ignored.
            - alpha: Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
            - gamma: Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples.
            - reduction: Specifies the reduction to apply to the output:
                ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
                ``'mean'``: the sum of the output will be divided by the number of
                elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
                and :attr:`reduce` are in the process of being deprecated, and in the meantime,
                specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``
        """
        
        super().__init__(reduction)
        self.focal = focal
        self.alpha = alpha
        self.gamma = gamma

        self.bce = nn.BCELoss(reduction="none")

    def forward(self, pred: Tensor, target: Tensor):
        """
        Args:
        ---
            - pred: unnormalized scores, shape: [N_obj, N_cls]
            - target: shape: [N_obj] for label encoding or [N_obj, N_cls] for one hot encoding
        """
        if target.ndim == 1:
            assert target.dtype == torch.long, \
                "Target with label encoding styple must be `long` type, but got {}".format(target.dtype)
            target = F.one_hot(target, num_classes=pred.size(-1) + 1)[..., :-1].float()
        else:
            assert target.ndim == 2, target.ndim

        probs = pred.sigmoid()
        loss = self.bce(probs, target)

        if self.focal:
            p_t = probs * target + (1 - probs) * (1 - target)
            loss = (1 - p_t) ** self.gamma * loss

            if self.alpha >= 0:
                alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
                loss = alpha_t * loss

        loss = loss.mean(1) # [N_obj, N_cls] -> [N_obj]

        return self._reduce(loss)
        

class ObjectnessLoss(ClassificationLoss):
    def forward(self, pred: Tensor, target: Tensor):
        """
        Args:
        ---
            - pred: unnormalized scores, shape: [N_obj, 1]
            - target: shape: [N_obj, 1]
        """
        return super().forward(pred, target)


class BoundingBoxLoss(Loss):
    def __init__(self, l1_weight: float = 1., giou_weight: float = 1., reduction: str = "mean") -> None:
        super().__init__(reduction)
        assert l1_weight > 0.
        assert giou_weight >= 0.

        self.l1_weight = l1_weight
        self.giou_weight = giou_weight

    def forward(self, pred: Tensor, target: Tensor):
        """
        Args:
        ---
            - pred: shape: [N_obj, 4]
            - target: shape: [N_obj, 4]

            box format: [cx, cy, w, h]
        """
        # l1 norm [N_obj, 4] -> [N_obj]
        dist_loss = F.l1_loss(pred, target, reduction=self.reduction).sum(-1)
        giou_loss = 1. - torch.diag(
            generalized_box_iou(
                box_cxcywh_to_xyxy(pred),
                box_cxcywh_to_xyxy(target)
            )
        ) # [N_obj]
        loss = self.l1_weight * dist_loss + self.giou_weight * giou_loss # [N_obj]
        
        return self._reduce(loss)


class CollectiveLoss(Loss):
    def __init__(self,
                 focal: bool = False,
                 focal_alpha: float = 0.25,
                 cls_loss_weight: float = 1.,
                 box_l1_loss_weight: float = 1.,
                 box_giou_loss_weight: float = 1.,
                 obj_loss_weight: float = 1.,
                 ) -> None:
        super().__init__(reduction="none")

        self.cls_loss_weight = cls_loss_weight
        self.box_l1_loss_weight = box_l1_loss_weight
        self.box_giou_loss_weight = box_giou_loss_weight
        self.obj_loss_weight = obj_loss_weight

        self.cls_loss = ClassificationLoss(focal, focal_alpha, reduction=self.reduction)
        self.box_loss = BoundingBoxLoss(box_l1_loss_weight, box_giou_loss_weight, self.reduction)
        self.obj_loss = ObjectnessLoss(focal, focal_alpha, reduction=self.reduction)

        self.loss_matrices: Tensor = None
        self.num_objs: Tensor = None

    def _forward_tracks(self, pred: BatchedOutputs, targets: list[GroundTruth], matched_track_results: list[tuple[Tensor, Tensor, Tensor, Tensor]] = None):
        device = pred.detection_boxes.device
        curr_frame_idx = targets[0].num_frames + pred.iter_index - 1

        if not pred.has_track_queries:
            return ((torch.empty([0, pred.num_classes + 1]).float().to(device), torch.empty(0).long().to(device)),\
                (torch.empty([0, 1]).float().to(device), torch.empty([0, 1]).float().to(device)),\
                (torch.empty([0, 4]).float().to(device), torch.empty([0, 4]).float().to(device))), torch.zeros(pred.batch_size).float().to(device)
        
        assert pred.track_boxes.shape[:2] == pred.track_padding_mask.shape # [B, N_t + N_fp + N_pad], N_t = N_exist + N_disapp

        # init predictions and ground truths placeholders
        batched_labels = torch.full_like(pred.track_padding_mask, pred.num_classes).long().to(device) # [B, N_t + N_fp + N_pad]
        batched_target_objness = torch.zeros_like(pred.track_objs).float().to(device) # [B, N_t + N_fp + N_pad, 1]
        batched_target_boxes = torch.zeros_like(pred.track_boxes).float().to(device) # [B, N_t + N_fp + N_pad, 4]

        batched_keep = pred.track_padding_mask > -2 # 0: tracks, -1: fp tracks, -2: padded tracks, drop padded tracks [B, N_t + N_fp] 
        batched_boxes_keep = torch.zeros_like(pred.track_padding_mask).bool().to(device) # [B, N_t + N_fp + N_pad]

        num_existing_track_objs = torch.zeros(pred.batch_size).float().to(device)

        for index in range(pred.batch_size): # batch index
            # [N_t], [N_exist], [N_t], [N_t]
            hs_indices, existing_gt_indices, gt_ids, keep = matched_track_results[index] # from last Hungarian matching results
            annos = targets[index].get_annotations(curr_frame_idx) # [N_objs, 7]
            existing_hs_indices = hs_indices[keep] # pick out disappeared tracks # [N_exist]
            num_existing_track_objs[index] = float(existing_hs_indices.numel())

            # existing tracks: supervise cls, durational occurence, reg
            # disappeared and fp tracks: supervise cls and durational occurence
            batched_labels[index, existing_hs_indices] = annos[existing_gt_indices][:, 0].long() # [N_exist]
            batched_target_objness[index, hs_indices] = targets[index].get_appearence(pred.iter_index, gt_ids).float() # [N_t, 1]
            batched_target_boxes[index, existing_hs_indices] = annos[existing_gt_indices][:, 2:6] # [N_exist, 4]
            batched_boxes_keep[index, existing_hs_indices] = True # set existing status to corresponding indices

        # cls and durational occurence: mask out padded tracks, box: mask out all negative and padded ones
        batched_logits, batched_labels = [item[batched_keep] for item in [pred.track_logits, batched_labels]] # [B * (N_t_i + N_fp_i), ...], i: batch index
        batched_pred_objness, batched_target_objness = [item[batched_keep] for item in [pred.track_objs, batched_target_objness]] # [B * (N_t_i + N_fp_i), ...]
        batched_pred_boxes, batched_target_boxes = [item[batched_boxes_keep] for item in [pred.track_boxes, batched_target_boxes]] # [B * N_exist_i, ...]

        return (
            (batched_logits, batched_labels), (batched_pred_objness, batched_target_objness), (batched_pred_boxes, batched_target_boxes)
        ), num_existing_track_objs

    def _forward_detections(self, pred: BatchedOutputs, targets: list[GroundTruth], matched_detect_results: list[tuple[Tensor, Tensor, Tensor]] = None):
        device = pred.detection_boxes.device
        curr_frame_idx = targets[0].num_frames + pred.iter_index - 1

        # init predictions and ground truths placeholders
        batched_labels = torch.full(pred.detection_logits.shape[:2], pred.num_classes).long().to(device) # [B, N]
        batched_target_objness = torch.zeros_like(pred.detection_objs).float().to(device) # [B, N, 1]
        batched_target_boxes = torch.zeros_like(pred.detection_boxes).float().to(device) # [B, N, 4]

        batched_boxes_keep = torch.zeros(pred.detection_boxes.shape[:2]).bool().to(device) # [B, N]
        num_new_detections = torch.zeros(pred.batch_size).float().to(device)

        for index in range(pred.batch_size):
            hs_indices, gt_indices, _ = matched_detect_results[index] # [N_d], [N_d]
            annos = targets[index].get_annotations(curr_frame_idx) # [N_obj, 7]
            
            # positives: supervise cls, durational occurence, reg
            # negatives: supervise cls and durational occurence
            batched_labels[index, hs_indices] = annos[gt_indices, 0].long() # [N_d]
            batched_target_objness[index, hs_indices] = 1. # [N_d, 1]
            batched_target_boxes[index, hs_indices] = annos[gt_indices, 2:6] # [N_d, 4]
            batched_boxes_keep[index, hs_indices] = True # [N_d]

            num_new_detections[index] = float(hs_indices.numel())
        
        return (
            (pred.detection_logits.flatten(0, 1), batched_labels.flatten(0, 1)), # [B*N, ...]
            (pred.detection_objs.flatten(0, 1), batched_target_objness.flatten(0, 1)), # [B*N, ...]
            (pred.detection_boxes[batched_boxes_keep], batched_target_boxes[batched_boxes_keep]) # mask out unmatched detections, [B * N_d_i, ...], i: batch index
        ), num_new_detections
    
    def reset(self):
        self.loss_matrices: Tensor = None
        self.num_objs: Tensor = None
    
    def forward(self, 
                pred: BatchedOutputs,
                targets: list[GroundTruth],
                matched_detect_results: list[tuple[Tensor, Tensor, Tensor]] = None,
                matched_track_results: list[tuple[Tensor, Tensor, Tensor, Tensor]] = None
                ):
        """
        Args:
        ---
            - pred: a `BatchedOutput` data structure
            - target: a list of `GroundTruth` data structure
            - matched_detect_results: new dectections with matched gt indices from `Matcher`
            - matched_track_results: inherited tracks with gt indices, ids from `Matcher`
        """
        device = pred.detection_boxes.device

        track_pred_gt_pairs, batched_num_tracks = self._forward_tracks(pred, targets, matched_track_results)
        detection_pred_gt_pairs, batched_num_detections = self._forward_detections(pred, targets, matched_detect_results)

        # calculate sizes of each batch as loss is compputed batch-wise with no reduction,
        # size list is used for spliting loss matrices
        # 1. get detection sizes
        cls_sizes = [pred.num_detect_queries] * pred.batch_size
        reg_sizes = batched_num_detections.long().tolist()

        # 2. get track sizes
        if pred.has_track_queries:
            _track_cls_sizes = (pred.track_padding_mask > -2).long().sum(-1).tolist() # [B]
            _track_reg_sizes = batched_num_tracks.long().tolist() # [B]
            cls_sizes += _track_cls_sizes # [2B]
            reg_sizes += _track_reg_sizes # [2B]

        # compute unreduced loss
        batched_losses = torch.zeros(3, len(cls_sizes)).float().to(device=device) # [3, B] or [3, 2B], 3: cls, obj, box
        for index, ((track_preds, track_gts), (detect_preds, detect_gts), loss_fn, loss_weight, split_size) in enumerate(
            zip(
                track_pred_gt_pairs, 
                detection_pred_gt_pairs, 
                [self.cls_loss, self.obj_loss, self.box_loss], # loss functions
                [self.cls_loss_weight, self.obj_loss_weight, 1.], # loss weights
                [cls_sizes, cls_sizes, reg_sizes], # splitting size
            )
        ):
            preds = torch.cat([track_preds, detect_preds], dim=0)
            gts = torch.cat([track_gts, detect_gts], dim=0)
            # [B * N_t_i + B*N, ...] | [B * N_exist_i + B * N_d_i, ...], i: batch index
            unreduced_loss = loss_weight * loss_fn.forward(preds, gts)
            # split to loss of individual samples then sum up, [B] or [2B]
            unreduced_loss = torch.stack([loss_unit.sum() for loss_unit in unreduced_loss.flatten().split(split_size)])
            batched_losses[index] = unreduced_loss

        if pred.has_track_queries: # has track loss, then loss = track_loss + det_loss, loss shape: [3, 2B]
            # [:, :B]: detection losses, [:, B:]: track losses, final shape: [3, B]
            batched_losses = batched_losses[:, :pred.batch_size] + batched_losses[:, pred.batch_size:]

        return batched_losses, batched_num_tracks + batched_num_detections
    
    def update(self, 
                pred: BatchedOutputs,
                targets: list[GroundTruth],
                matched_detect_results: list[tuple[Tensor, Tensor, Tensor]] = None,
                matched_track_results: list[tuple[Tensor, Tensor, Tensor, Tensor]] = None,
                ):
        """
        Calculate loss of one particular iterations and then collect it for 
            computing reduced loss.
        """
        #loss matrix: [3, B]
        loss_matrices, batched_num_objs = self.forward(pred, targets, matched_detect_results, matched_track_results)

        if pred.has_aux_outputs and self.training:
            for aux_pred in pred.aux_outputs:
                aux_loss, aux_num_objs = self.forward(aux_pred, targets, matched_detect_results, matched_track_results)

                loss_matrices += aux_loss
                batched_num_objs += aux_num_objs

        if self.num_objs is None:
            self.num_objs = batched_num_objs
            self.loss_matrices = loss_matrices
        else:
            self.num_objs += batched_num_objs # [B]
            self.loss_matrices += loss_matrices # [B]

        return loss_matrices, batched_num_objs
    
    def unsummed_result(self):
        """
        Return a loss vector that contains classification loss, objness loss and regression loss (un-summed).\\
        shape: [3]
        """
        # [3, B] div [1, B] -> [3]
        return self.loss_matrices.div(self.num_objs.view(1, -1)).mean(-1)
    
    def result(self):
        # [3, B] -> [B] -> ... -> [1]
        return self.loss_matrices.sum(0).div(self.num_objs).mean()

