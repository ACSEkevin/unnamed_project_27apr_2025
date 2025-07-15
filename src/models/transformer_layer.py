import torch

from torch import nn, Tensor, einsum
from torch.nn import functional as F
from typing import Optional
from einops import rearrange


class SpatialTemporalEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 8, dim_feedforward: int = 2048, dropout=0.1,
                 activation="relu", use_temporal_attn: bool = True):
        super().__init__()
        self.spatial_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        if use_temporal_attn:
            self.temporal_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
            self.temporal_norm = nn.LayerNorm(d_model)
            self.temporal_dropout = nn.Dropout(dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.spatial_norm = nn.LayerNorm(d_model)
        self.ffn_norm = nn.LayerNorm(d_model)

        self.spatial_dropout = nn.Dropout(dropout) 
        self.ffn_dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        self.use_temporal_attn = use_temporal_attn

    def with_pos_embed(self, tensor: Tensor, pos: Optional[Tensor]) -> Tensor:
        return tensor if pos is None else tensor + pos

    def forward(self,
                src: Tensor,
                src_mask: Tensor = None,
                src_key_padding_mask: Tensor = None,
                spatial_pos: Tensor = None,
                temporal_pos: Tensor = None):
        """
            src: [T, L, B, D]
            mask: [T, L, D]
        """
        T, L, B, D = src.shape
        src_pos = self.with_pos_embed(src, spatial_pos)
        q = k = src_pos.permute(1, 0, 2, 3).flatten(1, 2) # [L, TxB, D]
        v = src.permute(1, 0, 2, 3).flatten(1, 2)

        src2 = self.spatial_attn(q, k, value=v, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0] # attn
        src = v + self.spatial_dropout(src2) # shortcut
        src = self.spatial_norm(src) # norm

        if self.use_temporal_attn:
            # [L, TxB, D] -> [T, LxB, D]
            src = src.contiguous().view(L, T, B, D).transpose(0, 1)
            src_pos = self.with_pos_embed(src, temporal_pos)
            q = k = src_pos.flatten(1, 2)
            v = src.flatten(1, 2)
            src2 = self.temporal_attn.forward(q, k, value=v)[0]
            src = v + self.temporal_dropout(src2)
            src = self.temporal_norm(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.ffn_dropout(src2)
        src = self.ffn_norm(src)

        return src.contiguous().view(T, L, B, D)
      

class SpatialDecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation="relu"):
        super().__init__()
        self.query_self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.spatial_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)

        self.temporal_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.temporal_norm = nn.LayerNorm(d_model)
        self.temporal_dropout = nn.Dropout(dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.sa_norm = nn.LayerNorm(d_model)
        self.spatial_norm = nn.LayerNorm(d_model)
        self.ffn_norm = nn.LayerNorm(d_model)

        self.sa_dropout = nn.Dropout(dropout)
        self.spatial_dropout = nn.Dropout(dropout)
        self.ffn_dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor: Tensor, pos: Tensor = None):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt: Tensor, memory: Tensor,
                     tgt_mask: Tensor = None,
                     memory_mask: Tensor = None,
                     tgt_key_padding_mask: Tensor = None,
                     memory_key_padding_mask: Tensor = None,
                     pos: Tensor = None,
                     query_spatial_pos: Tensor = None,
                     query_temporal_pos: Tensor = None):
        """
        Args:
        ---
            - tgt: [S, B, D]
            - memory: [T, L, B, D]
            - tgt_mask: target attention mask, for query self-attention, [S, S]
            - memory_mask: mask for cross-attention, [S, L]
            - tgt_key_padding_mask: [B, S]
            - memory_key_padding_mask: [BT, HW]
            - query_spatial_pos: [N, B, D]
            - query_tempoal_pos: [T, N, B, D]
        """

        N, B, D = tgt.shape
        T, L, B, D = memory.shape

        tgt_pos = self.with_pos_embed(tgt, query_spatial_pos)

        tgt2 = self.query_self_attn(tgt_pos, tgt_pos, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.sa_dropout(tgt2)
        tgt = self.sa_norm(tgt) # [N, B, D]

        tgt = tgt.unsqueeze(0).repeat(T, 1, 1, 1) # [T, N, B, D]

        v = memory.transpose(0, 1).flatten(1, 2)
        q = tgt.transpose(0, 1).flatten(1, 2) # [N, TxB, D]
        k = self.with_pos_embed(memory, pos)
        k = k.transpose(0, 1).flatten(1, 2) # [L, TxB, D]
        tgt2 = self.spatial_attn(query=q, key=k, value=v, 
                                 attn_mask=memory_mask,
                                 key_padding_mask=memory_key_padding_mask)[0] # [N, TxB, D]
        tgt = q + self.spatial_dropout(tgt2)
        tgt = self.spatial_norm(tgt) # [N, TxB, D]

         # [T, N, B, D] -> [T, NxB, D]
        tgt = tgt.contiguous().view(N, T, B, D).transpose(0, 1).flatten(1, 2)
        tgt_with_pos = self.with_pos_embed(tgt, query_temporal_pos.flatten(1, 2)) # temporal pos encoding
        tgt2 = self.temporal_attn.forward(query=tgt_with_pos, key=tgt_with_pos, value=tgt)[0]
        tgt = tgt + self.temporal_dropout(tgt2)
        tgt = self.temporal_norm(tgt)[-1] # [T, NxB, D] -> [NxB, D]

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt)))) # [NxB, D]
        tgt = tgt + self.ffn_dropout(tgt2)
        tgt = self.ffn_norm(tgt)

        return tgt.contiguous().view(N, B, D)
    

class TrajectoryAttention(nn.Module):
    """
    Trajectory attention, copied and modified from 
    [MotionFormer](https://github.com/facebookresearch/Motionformer/blob/main/slowfast/models/vit_helper.py).
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., use_original_code=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # A typo in the original code meant that the value tensors for the temporal
        # attention step were identical to the input instead of being multiplied by a
        # learned projection matrix (v = x rather than v = Wx). The original code is
        # kept to facilitate replication, but is not recommended.
        self.use_original_code = use_original_code

    @staticmethod
    def _qkv_attn_op(q: Tensor, k: Tensor, v: Tensor):
        sim = einsum('b i d, b j d -> b i j', q, k)
        attn = sim.softmax(dim=-1)
        out = einsum('b i j, b j d -> b i d', attn, v)
        return out

    def forward(self, x, seq_len=196, num_frames=8, num_landmarks=128):
        B, N, C = x.shape
        P = seq_len
        F = num_frames
        h = self.num_heads

        # project x to q, k, v vaalues
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        # Reshape: 'b n (h d) -> (b h) n d'
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        # remove CLS token from q, k, v
        (cls_q, q_), (cls_k, k_), (cls_v, v_) = map(
            lambda t: (t[:, 0:1], t[:, 1:]), (q, k, v))

        # let CLS token attend to key / values of all patches across time and space
        cls_out = self._qkv_attn_op(cls_q * self.scale, k, v)
        cls_out = rearrange(cls_out, f'(b h) f d -> b f (h d)', f=1, h=h)

        
        # Using full attention
        q_dot_k = q_ @ k_.transpose(-2, -1)
        q_dot_k = rearrange(q_dot_k, 'b q (f n) -> b q f n', f=F)
        space_attn = (self.scale * q_dot_k).softmax(dim=-1)
        attn = self.attn_drop(space_attn)
        v_ = rearrange(v_, 'b (f n) d -> b f n d', f=F, n=P)
        x = torch.einsum('b q f n, b f n d -> b q f d', attn, v_)

        # Temporal attention: query is the similarity-aggregated patch
        x = rearrange(x, '(b h) s f d -> b s f (h d)', b=B)
        x_diag = rearrange(x, 'b (g n) f d -> b g n f d', g=F)
        x_diag = torch.diagonal(x_diag, dim1=-4, dim2=-2)
        x_diag = rearrange(x_diag, f'b n d f -> b (f n) d', f=F)
        q2 = self.proj_q(x_diag)
        k2, v2 = self.proj_kv(x).chunk(2, dim=-1)
        q2 = rearrange(q2, f'b s (h d) -> b h s d', h=h)
        q2 *= self.scale
        k2, v2 = map(
            lambda t: rearrange(t, f'b s f (h d) -> b h s f d', f=F,  h=h), (k2, v2))
        attn = torch.einsum('b h s d, b h s f d -> b h s f', q2, k2)
        attn = attn.softmax(dim=-1)
        if self.use_original_code:
            x = rearrange(x, f'b s f (h d) -> b h s f d', f=F,  h=h)
            x = torch.einsum('b h s f, b h s f d -> b h s d', attn, x)
        else:
            x = torch.einsum('b h s f, b h s f d -> b h s d', attn, v2)
        x = rearrange(x, f'b h s d -> b s (h d)')

        # concat back the cls token
        x = torch.cat((cls_out, x), dim=1)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


__all__ = [
    "SpatialTemporalEncoderLayer",
    "SpatialDecoderLayer",
    "TrajectoryAttention"
]
