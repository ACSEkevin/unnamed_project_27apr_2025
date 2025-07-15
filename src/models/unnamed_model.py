from .transformer import SpatialTemporalTransformer
from .position_encoding import *
from .head import *
from .backbone import *
from ..utils.misc import nested_tensor_from_tensor_list, NestedTensor
from ..utils.api import BatchedOutputs

from torch import nn, Tensor
from typing import Union

import torch


class UnnamedModel(nn.Module):
    def __init__(self, num_frames: int, num_classes: int, 
                 d_model: int = 256,
                 num_heads: int = 8,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6,
                 dim_ffn: int = 2048,
                 num_queries: int = 100,
                 update_track_query_pos: bool = True,
                 update_dropout: float = 0.1,
                 backbone_name: str = "resnet50",
                 freeze_backbone: bool = False,
                 backbone_dilation: bool = False,
                 dropout: float = 0.1,
                 aux_output: bool = False, 
                 ) -> None:
        super().__init__()
        self.name = self.__class__.__name__
        self.num_frames = num_frames
        self.num_classes = num_classes
        self.d_model = d_model
        self.num_queries = num_queries
        self.aux_output = aux_output
        self.update_track_query_pos = update_track_query_pos

        # pos embeddings
        assert d_model % 2 == 0, "Hidden dimension must be divisible by 2."
        self.spatial_pos_embed = PositionEmbeddingFixed2D(d_model // 2, normalize=True)
        self.temporal_pos_embed = PositionEmbeddingLearnedTemporal(num_frames, d_model)

        self.queries = nn.Embedding(num_queries, d_model)
        self.query_spatial_pos_embed = nn.Embedding(num_queries, d_model)
        self.query_temporal_pos_embed = PositionEmbeddingLearnedTemporal(num_frames, d_model, False)

        if update_track_query_pos:
            self.query_update_mlp = MLP(d_model, dim_ffn, d_model, 2, update_dropout)
            self.query_update_norm = nn.LayerNorm(d_model)

        # archs
        self.backbone = Backbone(backbone_name, not freeze_backbone, False, backbone_dilation)
        ## backbone proj to d_model
        self.input_proj = nn.Conv2d(self.backbone.num_channels, d_model, kernel_size=1)
        self.transformer = SpatialTemporalTransformer(
            num_frames, d_model, nhead=num_heads,
            num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, dim_feedforward=dim_ffn,
            dropout=dropout, activation="relu", enc_use_temporal_attn=True,
            dec_return_intermediate=aux_output
        )
        self.head = Head(d_model, d_model, num_classes)
        
        self.feat_queue: NestedTensor = None

    def _validate_input(self, x: Union[Tensor, NestedTensor, list[Tensor]]):
        assert isinstance(x, (Tensor, NestedTensor, list)), type(x)

        # FIXME: size check is not applied on NestedTensor 
        if isinstance(x, NestedTensor):
            return x
        
        if isinstance(x, list):
            time_dim = 0
            sample = x[0]
        else:
            time_dim = 1
            sample = x

        if self.feat_queue is not None:
            decode_size = 1 # single-image inference
            err_msg = "after"
        else: # not pre-filled
            decode_size = self.num_frames
            err_msg = "before"

        assert x.size(time_dim) == decode_size,\
            "{} frame(s) must be passed {} `pre-fill` stage, but got {}, size: {}".format(
            decode_size, err_msg, sample.size(time_dim), sample.size()
        )

        return x
    
    def _preproc_inputs(self, x: Union[Tensor, NestedTensor, list[Tensor]]) -> NestedTensor:
        """
        convert inputs to a standard `NestedTensor` type. This includes padding and masks building.
        """
        x = self._validate_input(x)

        if isinstance(x, list):
            tensor_list = []
            for clip in x: # [T, 3, H, W]
                tensor_list.extend(
                    [clip[i] for i in range(clip.size(0))]
                )
            x = nested_tensor_from_tensor_list(tensor_list)
        elif isinstance(x, Tensor):
            x = nested_tensor_from_tensor_list(x.flatten(0, 1))

        return x
    
    def _update_track_query_pos_embed(self, x: Tensor) -> Tensor:
        x1 = self.query_update_mlp.forward(x)
        x = x + x1 # shorcut
        x = self.query_update_norm.forward(x)

        return x

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class: Tensor, outputs_coord: Tensor, outputs_obj: Tensor):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [BatchedOutputs(
                    {'pred_logits': a, 'pred_boxes': b, 'pred_objs': c, 
                     "hs": torch.empty([0, self.d_model], device=a.device).float(), 
                     "query_spatial_pos": torch.empty([0, self.d_model], device=a.device).float()}, 
                    self.num_queries
                )
                for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_obj[:-1])]

    def build_position_embeddings(self, features: NestedTensor):
        spatial_pos = self.spatial_pos_embed.forward(features).unflatten(0, [-1, self.num_frames]) # [B, T, D, H, W]
        query_spatial_pos = self.query_spatial_pos_embed.weight.unsqueeze(0).repeat(spatial_pos.size(0), 1, 1) # [B, Nq, D] 

        features.unflatten_by_seq_(self.num_frames) # [B, T, ...]
        temporal_pos = self.temporal_pos_embed.forward(features) # [B, T, D, H, W]
        query_temporal_pos = self.query_temporal_pos_embed.forward(features) # [B, T, D]

        return spatial_pos, temporal_pos, query_spatial_pos, query_temporal_pos

    def extract_and_project(self, x: NestedTensor) -> NestedTensor:
        outs = self.backbone.forward(x)
        feats = [feat for _, feat in outs.items()][-1] # [BT, C, H, W], [BT, H, W]
        feats.tensors = self.input_proj.forward(feats.tensors) # [BT, D, H, W]

        return feats

    def forward(self,
                x: Union[Tensor, NestedTensor, list[Tensor]],
                track_queries_with_pos: NestedTensor = None):
        """
        Args:
        ---
            - x: batched video clip with a shape of [B, T, 3, H, W] or a `NestedTensor`, `T=1` 
                after "prefill" stage.
            - track_querires_with_pos: a `NestedTensor` containing:
                - tensors: track queries from last detection, shape [B, Nt, 2*D], [..., :D]: track queries, 
                [..., D:]: corresponding spatial position embeddings
                - mask: tracks queries padding mask in which element `1` denotes should-mask and vice versa.
                    shape: [B, Nt, D]
        """
        x = self._preproc_inputs(x)
        if self.feat_queue is not None: # already prefilled, here as a pre-check
            if track_queries_with_pos is not None:
                assert track_queries_with_pos.tensors.size(2) == self.d_model * 2, \
                    "Expected track queries last size == {}, but got {}".format(
                    self.d_model * 2, track_queries_with_pos.tensors.size(2)
                )
            
        x = self.extract_and_project(x) # [BT, D, H, W] / [B*1, D, H, W]

        if self.feat_queue is None: # not prefilled
            self.feat_queue = x
        else: # prefilled, drop oldest frame and concat the lastest
            if self.feat_queue.tensors.ndim == 4: # [BT, D, H, W]
                self.feat_queue.unflatten_by_seq_(self.num_frames) # [B, T, ...]

            self.feat_queue = self.feat_queue[:, 1:] # drop first [B, T-1, ...]
            self.feat_queue.flatten_bs_seq_() # [B*(T - 1), ...]
            self.feat_queue = self.feat_queue.cat(x, dim=0) # [BT, ...]

        # [B, T, D, H, W], [B, T, D, H, W], [B, Nq, D], [B, T, D]
        spatial_pos, temporal_pos, query_spatial_pos, query_temporal_pos = self.build_position_embeddings(self.feat_queue)
        tgts = self.queries.weight.unsqueeze(0).repeat(spatial_pos.size(0), 1, 1) # [B, Nq, D]
        tgt_mask = None # key padding mask for query self attn

        if track_queries_with_pos is not None: # concat queries with track queries
            track_queries, track_query_spatial_pos = track_queries_with_pos.tensors.split(self.d_model, dim=-1) # [B, N_t, D], [...]
            
            if self.update_track_query_pos:
                track_query_spatial_pos = self._update_track_query_pos_embed(track_query_spatial_pos)

            tgts = torch.cat([tgts, track_queries], dim=1) # [B, N, D], N = Nq + Nt
            query_spatial_pos = torch.cat([query_spatial_pos, track_query_spatial_pos], dim=1) # [B, N, D]

            if track_queries_with_pos.mask is not None: # [B, Nt]
                tgt_mask = torch.zeros(tgts.shape[:2]).bool().to(tgts.device) # [B, N], N = Nq + Nt
                tgt_mask[..., self.num_queries:] = track_queries_with_pos.mask > -2. # -2: padded queries
        
        # [n_inter, B, N, D]
        hidden_state, _ = self.transformer.forward(
            self.feat_queue.tensors, tgts, # src tgt
            spatial_pos, temporal_pos, query_spatial_pos, query_temporal_pos, # pos embeddings
            self.feat_queue.mask, tgt_mask # masks
        )
        # [B, N, N_cls + 1], [B, N, 4], [B, N, 1]
        cls_out, box_out, obj_out = self.head.forward(hidden_state)
        outs = {
            'pred_logits': cls_out[-1], 'pred_boxes': box_out[-1],
            'pred_objs': obj_out[-1], 'hs': hidden_state[-1]
        }
        outs["query_spatial_pos"] = query_spatial_pos

        if self.aux_output:
            outs['aux_outputs'] = self._set_aux_loss(cls_out, box_out, obj_out)  

        wrapped_outs = BatchedOutputs(outs, self.num_queries)
        if track_queries_with_pos is not None:
            wrapped_outs.set_track_padding_mask(track_queries_with_pos.mask)

        return wrapped_outs
    
    def reset_prefill(self):
        self.feat_queue: NestedTensor = None


__all__ = [
    "UnnamedModel"
]



