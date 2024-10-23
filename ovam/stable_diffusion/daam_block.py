import math
from typing import TYPE_CHECKING, Iterable, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn

from ..base.daam_block import DAAMBlock
from ..utils.attention_ops import (
    ActivationTypeVar,
    AggregationTypeVar,
    apply_activation,
    apply_aggregation,
)

if TYPE_CHECKING:
    from ..base.attention_storage import AttentionStorage

__all__ = ["CrossAttentionDAAMBlock"]


class CrossAttentionDAAMBlock(DAAMBlock):
    """This DaamBlock correspond to the attention block o a upsampling or
    downsampling layer of the UNet2DConditionModel
    """

    def __init__(
        self,
        to_k: nn.Linear,
        hidden_states: Union["AttentionStorage", Iterable["torch.Tensor"]],
        scale: float,
        heads: int,
        name: str,
        heads_activation: Optional["ActivationTypeVar"] = None,
        blocks_activation: Optional["ActivationTypeVar"] = None,
        heads_aggregation: "AggregationTypeVar" = "sum",
        heads_epochs_activation: Optional["ActivationTypeVar"] = None,
        heads_epochs_aggregation: "AggregationTypeVar" = "sum",
    ):
        super().__init__(hidden_states=hidden_states, name=name)

        self.to_k = to_k
        self.scale = scale
        self.heads = heads
        self.heads_activation = heads_activation
        self.blocks_activation = blocks_activation
        self.heads_aggregation = heads_aggregation
        self.heads_epochs_activation = heads_epochs_activation
        self.heads_epochs_aggregation = heads_epochs_aggregation

    def _compute_attention(self, query, key):
        """
        Monkey-patched version of :py:func:`.CrossAttention._attention` to capture attentions and aggregate them.

        Args:
            self (`CrossAttention`): pointer to the module.
            query (`torch.Tensor`): the query tensor.
            key (`torch.Tensor`): the key tensor.
            value (`torch.Tensor`): the value tensor.
        """
        # Cross attention matrix Wq*h x (Wk*X)^T
        attention_scores = torch.baddbmm(
            torch.empty(
                query.shape[0],
                query.shape[1],
                key.shape[1],
                dtype=query.dtype,
                device=query.device,
            ),
            query,
            key.transpose(-1, -2),
            beta=0,
            alpha=self.scale,
        )
        
        attention_probs = attention_scores.softmax(dim=-1)
        attention_probs = attention_probs.to(query.dtype)
        attention_probs = attention_probs.transpose(-1, -2)
        
        head_size = self.heads
        batch_size, seq_len, dim = attention_probs.shape
        attention_probs = attention_probs.reshape(batch_size // head_size, head_size, seq_len, dim)
        
        # query = self.reshape_batch_to_head_dim(query)
        # queries.append(query)

        # # Unravel the attention scores into a collection of heatmaps
        # # Unravel Based on the of the function `unravel_attn` of daam.trace
        # h = w = int(math.sqrt(attention_scores.size(1)))
        # # attention_scores = attention_scores.permute(
        # #     2, 0, 1
        # # )  # shape: (tokens, heads, h*w)

        # attention_scores = attention_scores.reshape(
        #     (attention_scores.size(0), attention_scores.size(1), h, w)
        # )  # shape: (tokens, heads, h, w)
        # attention_scores = attention_scores.permute(
        #     1, 0, 2, 3
        # ).contiguous()  # shape: (heads, tokens, h, w)

        return attention_probs  # shape: (batch, heads, tokens, dim)

    def reshape_heads_to_batch_dim(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        extra_dim = 1
        tensor = tensor.reshape(batch_size, seq_len * extra_dim, head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size * head_size, seq_len * extra_dim, dim // head_size)

        return tensor

    @torch.enable_grad()
    def forward(self, x):
        """Compute the attention for a given input x"""

        key = self.to_k(x)  # Shape: (batch, n_tokens, embedding_size = 1280)
        key = self.reshape_heads_to_batch_dim(key)  # Shape: (batch*n_heads, n_tokens, latent=160)

        heatmaps = []  #  List of heatmaps

        # Batch images can have different sizes and be stored offline
        # This loop is not vectorized for this reason
        for batch_image in self.hidden_states:
            #  TODO: This second loop can be vectorized with einsum
            attentions = []  #  List of heatmaps
            for query in batch_image:  #  ()
                attention = self._compute_attention(
                    query, key
                )  # shape: (heads, tokens, height, width)
                h = w = int(math.sqrt(attention.size(-1)))
                new_shape  = attention.shape[:-1] + (h, w) 
                attention =  attention.reshape(*new_shape)
                attentions.append(attention)
            
            # for head dimension and epoch dimension
            attention = torch.stack(
                attentions
            )  # Shape: (n_epochs, batch_size, heads, n_tokens, latent_size / factor, latent_size / factor)
            
            # Collapse epochs
            # Shape: (batch_size, heads, n_tokens, latent_size / factor, latent_size / factor)
            # attention = apply_activation(attention, self.heads_epochs_activation) # already performed within _compute_attention
            attention = apply_aggregation(attention, self.heads_epochs_aggregation, dim=0)
            
            # Collapse heads dimension
            # Shape: (batch_size, n_tokens, latent_size / factor, latent_size / factor)
            attention = apply_activation(attention, self.heads_activation)
            attention = apply_aggregation(attention, self.heads_aggregation, dim=1)
            

            # Shape: (batch, n_tokens, latent_size / factor, latent_size / factor)
            heatmaps.append(attention)

        # Shape (n_images, n_tokens, output_size, output_size)
        heatmaps = torch.cat(heatmaps, dim=0)

        # ids = torch.arange(len(x))
        return heatmaps[:, 1, :, :].unsqueeze(1)
