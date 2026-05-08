import torch
import torch.nn as nn

from typing import Any, Dict, List, Optional, Tuple
from diffusers.configuration_utils import ConfigMixin

from learning.tinymdm.modules import AdaLayerNormSingle, SinusoidalPositionalEmbedding, StableMotionDiTBlock

class TinyStableMotionDiTModel(nn.Module, ConfigMixin):
    def __init__(
        self,
        in_channels: int = 64,
        num_layers: int = 4,
        attention_head_dim: int = 64,
        num_attention_heads: int = 4,
        out_channels: int = 64,
        dropout: float = 0.,
        max_seq_len: int = 32,
    ):
        super().__init__()

        self.dropout = dropout
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        self.transformer_blocks = nn.ModuleList(
            [
                StableMotionDiTBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    dropout=self.dropout,
                )
                for i in range(num_layers)
            ]
        )

        self.scale_shift_table = nn.Parameter(torch.randn(1, 1, 2, self.inner_dim) / self.inner_dim**0.5)

        self.adaln_single = AdaLayerNormSingle(
            self.inner_dim, 
        )
        
        self.preprocess_conv = nn.Conv1d(self.in_channels, self.in_channels, 1, bias=False) # Why 1*1 conv, same as linear, feature fusion?
        self.proj_in = nn.Linear(self.in_channels, self.inner_dim, bias=False)
        self.proj_out = nn.Linear(self.inner_dim, self.out_channels, bias=False)
        self.postprocess_conv = nn.Conv1d(self.out_channels, self.out_channels, 1, bias=False)
        self.sequence_pos_encoder = SinusoidalPositionalEmbedding(self.inner_dim, max_seq_length=max_seq_len)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        timestep: torch.LongTensor = None,
        attention_mask: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> torch.FloatTensor:        
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.view(batch_size, -1, self.in_channels)
        hidden_states = hidden_states.transpose(1, 2)

        hidden_states = self.preprocess_conv(hidden_states) + hidden_states
        # (batch_size, dim, sequence_length) -> (batch_size, sequence_length, dim)
        hidden_states = hidden_states.transpose(1, 2)

        hidden_states = self.proj_in(hidden_states)

        time_hidden_states, embedded_timestep = self.adaln_single(timestep)
        
        # Add pos embedding
        hidden_states = self.sequence_pos_encoder(hidden_states)

        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=None, #cross_attention_hidden_states,
                encoder_attention_mask=None, #encoder_attention_mask,
                time_hidden_states=time_hidden_states,
            )

        hidden_states = self.proj_out(hidden_states)

        # (batch_size, sequence_length, dim) -> (batch_size, dim, sequence_length)
        hidden_states = hidden_states.transpose(1, 2)

        hidden_states = self.postprocess_conv(hidden_states) + hidden_states

        hidden_states = hidden_states.transpose(1, 2)

        hidden_states = hidden_states.reshape(batch_size, -1)

        return hidden_states
    

class CondTinyStableMotionDiTModel(nn.Module, ConfigMixin):
    def __init__(
        self,
        in_channels: int = 64,
        num_layers: int = 4,
        attention_head_dim: int = 64,
        num_attention_heads: int = 4,
        out_channels: int = 64,
        dropout: float = 0.,
        num_class: int = 0,
        cfg_dropout: float = 0.0, # Use dropout for CFG
        max_seq_len: int = 32,
    ):
        super().__init__()

        self.dropout = dropout
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        self.transformer_blocks = nn.ModuleList(
            [
                StableMotionDiTBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    dropout=self.dropout,
                )
                for i in range(num_layers)
            ]
        )

        self.scale_shift_table = nn.Parameter(torch.randn(1, 1, 2, self.inner_dim) / self.inner_dim**0.5)

        self.adaln_single = AdaLayerNormSingle(
            self.inner_dim, 
            use_additional_conditions=num_class > 0,
            num_class=num_class,
            cfg_dropout=cfg_dropout, # Use dropout for CFG
        )
        
        self.preprocess_conv = nn.Conv1d(self.in_channels, self.in_channels, 1, bias=False) # Why 1*1 conv, same as linear, feature fusion?
        self.proj_in = nn.Linear(self.in_channels, self.inner_dim, bias=False)
        self.proj_out = nn.Linear(self.inner_dim, self.out_channels, bias=False)
        self.postprocess_conv = nn.Conv1d(self.out_channels, self.out_channels, 1, bias=False)
        self.sequence_pos_encoder = SinusoidalPositionalEmbedding(self.inner_dim, max_seq_length=max_seq_len)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        timestep: torch.LongTensor = None,
        global_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.LongTensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> torch.FloatTensor:
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.view(batch_size, -1, self.in_channels)
        hidden_states = hidden_states.transpose(1, 2)

        hidden_states = self.preprocess_conv(hidden_states) + hidden_states
        # (batch_size, dim, sequence_length) -> (batch_size, sequence_length, dim)
        hidden_states = hidden_states.transpose(1, 2)

        hidden_states = self.proj_in(hidden_states)
        
        time_hidden_states, embedded_timestep = self.adaln_single(timestep, class_labels, **kwargs)
        
        # Add pos embedding
        hidden_states = self.sequence_pos_encoder(hidden_states)

        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=None, #cross_attention_hidden_states,
                encoder_attention_mask=None, #encoder_attention_mask,
                time_hidden_states=time_hidden_states,
            )

        hidden_states = self.proj_out(hidden_states)

        # (batch_size, sequence_length, dim) -> (batch_size, dim, sequence_length)
        hidden_states = hidden_states.transpose(1, 2)

        hidden_states = self.postprocess_conv(hidden_states) + hidden_states

        hidden_states = hidden_states.transpose(1, 2)

        hidden_states = hidden_states.reshape(batch_size, -1)

        return hidden_states