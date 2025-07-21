import torch, copy

from torch import nn, Tensor
from torch.nn import functional as F
from typing import Optional

from .transformer_layer import SpatialDecoderLayer, SpatialTemporalEncoderLayer


class SpatialTemporalTransformer(nn.Module):
    def __init__(self, 
                 num_frames: int = None,
                 d_model: int = 512, 
                 nhead: int = 8,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, 
                 dim_feedforward: int = 2048, 
                 dropout: float = 0.1,
                 activation: str = "relu",
                 enc_use_temporal_attn: bool = False,
                 dec_return_intermediate: bool = False):
        super().__init__()

        encoder_layer = SpatialTemporalEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, enc_use_temporal_attn)
        
        self.encoder = SpatialTemporalTransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = SpatialDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = SpatialTemporalTransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=dec_return_intermediate)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead
        self.num_frames: int = num_frames # store n_frames when training, not used
        self.return_intermediate = dec_return_intermediate

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, 
                src: Tensor,
                queries: Tensor, 
                spatial_pos_embed: Tensor,
                temporal_pos_embed: Tensor,
                query_spatial_embed: Tensor, 
                query_temporal_embed: Tensor = None, 
                src_mask: Tensor = None,
                tgt_mask: Tensor = None):
        """
        Args:
        ---
            - src: features [B, T, D, fH, fW]
            - queries: [B, N, D], decoder queries which must be passed as queries 
                are processed outside the transformer,
            - spatial_pos_embed: [B, T, D, fH, fW], with batch sizes means all batches 
                do not share one pos embedding due to different masks
            - temporal_pos_embed: [B, T, D, fH, fW], all batches can share one embedding.
            - query_spatial_embed: spatial query positional embedding, [B, N, D] 
            - query_temporal_embed: temporal query positional embedding, [B, T, D]
            - src_mask: key padding mask for images, [B, T, H, W]
            - tgt_mask: key padding mask in decoder query-related attention, [B, N]
        """
        B, T, D, fH, fW = src.shape
        src = src.flatten(3).permute(1, 3, 0, 2) # [T, HW, B, D]
        queries = queries.transpose(0, 1) # [N, B, D]
        spatial_pos_embed = spatial_pos_embed.flatten(3).permute(1, 3, 0, 2)
        temporal_pos_embed = temporal_pos_embed.flatten(3).permute(1, 3, 0, 2)

        query_spatial_embed = query_spatial_embed.transpose(0, 1) # [N, B, D]
        if query_temporal_embed is not None:
            N_query = query_spatial_embed.size(0)
            query_temporal_embed = query_temporal_embed.transpose(0, 1) # [T, B, D]
            query_temporal_embed = query_temporal_embed.unsqueeze(1).repeat(1, N_query, 1, 1) # [T, N, B, D]

        if src_mask is not None:
            src_mask = src_mask.flatten(0, 1).flatten(1, 2) # [BT, HW]

        memory = self.encoder(
            src,
            src_key_padding_mask=src_mask, 
            spatial_pos=spatial_pos_embed,
            temporal_pos=temporal_pos_embed
        ) # [T, HW, B, D]
        hs = self.decoder(queries, memory, 
                                  memory_key_padding_mask=src_mask,
                                  tgt_key_padding_mask=tgt_mask,
                                  pos=spatial_pos_embed, 
                                  query_spatial_pos=query_spatial_embed,
                                  query_temporal_pos=query_temporal_embed
                                ) # [N_inter, N, B, D]
        if not self.return_intermediate:
            hs = hs.unsqueeze(0)
        
        return hs.transpose(1, 2), memory.permute(2, 0, 3, 1).contiguous().view(B, T, D, fH, fW)


class SpatialTemporalTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer: nn.Module, num_layers: int, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self,
                src: Tensor,
                mask: Tensor = None,
                src_key_padding_mask: Tensor = None,
                spatial_pos: Tensor = None,
                temporal_pos: Tensor = None):
        output = src

        for layer in self.layers:
            output = layer(
                output, 
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask, 
                spatial_pos=spatial_pos,
                temporal_pos=temporal_pos
            )

        if self.norm is not None:
            output = self.norm(output)
            
        return output


class SpatialTemporalTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer: nn.Module, num_layers: int, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_spatial_pos: Optional[Tensor] = None,
                query_temporal_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos,
                           query_spatial_pos=query_spatial_pos,
                           query_temporal_pos=query_temporal_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def build_spatial_temporal_transformer(args):
    return SpatialTemporalTransformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        enc_use_temporal_attn=False,
        dec_return_intermediate=True,
    )


__all__ = [
    "SpatialTemporalTransformerEncoder",
    "SpatialTemporalTransformerDecoder",
    "SpatialTemporalTransformer",
    "build_spatial_temporal_transformer"
]

