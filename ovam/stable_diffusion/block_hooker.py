from typing import TYPE_CHECKING, List

import torch

from ..base.block_hooker import BlockHooker
from .daam_block import CrossAttentionDAAMBlock


if TYPE_CHECKING:
    from diffusers.models.attention import CrossAttention

__all__ = ["CrossAttentionHooker"]


class CrossAttentionHooker(BlockHooker):
    def __init__(
        self,
        module: "CrossAttention",
        name: str,
        store_unconditional_hidden_states: bool = False,
        store_conditional_hidden_states: bool = True,
    ):
        super().__init__(module=module, name=name)
        self._current_hidden_state: List["torch.tensor"] = []
        self.store_conditional_hidden_states = store_conditional_hidden_states
        self.store_unconditional_hidden_states = store_unconditional_hidden_states

    def _hooked_forward(
        hk_self: "BlockHooker",
        attn: "CrossAttention",
        hidden_states: "torch.Tensor",
        encoder_hidden_states: "torch.Tensor",
        attention_mask: "torch.Tensor",
        temb=None,
        **kwargs,
    ):
        """
        Path: diffusers/models/unet_2d_blocks.KAttnetionBlock.forward
        Hooked forward of the cross attention module.

        Stores the hidden states and perform the original attention.
        """
        
        # return hk_self.monkey_super("forward", hidden_states, **kwargs)
        
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        # shape (batch * heads * 2, dim, 77)
        batch_size = attention_probs.shape[0] // 2
        if hk_self.store_unconditional_hidden_states:
            hk_self._current_hidden_state.append(attention_probs[:batch_size])
        if hk_self.store_conditional_hidden_states:
            assert attention_probs.shape[0] > 1
            hk_self._current_hidden_state.append(attention_probs[batch_size:])
            
        return hidden_states
    
    
    def reshape_batch_to_head_dim(self, tensor):
        head_size = self.module.heads
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        return tensor

    @torch.enable_grad()
    def store_hidden_states(self) -> None:
        """Stores the hidden states in the parent trace"""
        if not self._current_hidden_state:
            return

        queries = []  # This loop can be vectorized, but has a small impact
        # Thus it is not executed during the optimization process
        
        
        for c in self._current_hidden_state:             
            query = c.transpose(-1, -2)
            query = self.reshape_batch_to_head_dim(query)
            queries.append(query)

        # n_epochs x heads x inner_dim x (latent_size = 64)
        current_hidden_states = torch.stack(queries)
        
        self.hidden_states.store(current_hidden_states)

        self._current_hidden_state = []  # Clear the current hidden states

    def daam_block(self, **kwargs) -> "CrossAttentionDAAMBlock":
        """Builds a DAAMBlock with the current hidden states.

        Arguments
        ---------
        **kwargs:
            Arguments passed to the `DAAMBlock` constructor.
        """

        return CrossAttentionDAAMBlock(
            to_k=self.module.to_k,
            hidden_states=self.hidden_states,
            scale=self.module.scale,
            heads=self.module.heads,
            name=self.name,
            **kwargs,
        )
