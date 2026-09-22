import torch
import torch.nn as nn
import re


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')


def build_mm_qformer(num_query_token=448, delay_load=False, **kwargs):
    mm_hidden_size = 1024
    cross_attention_freq = 1
    num_hidden_layers = 2
    from .Qformer import BertLMHeadModel, BertConfig
    encoder_config = BertConfig.from_pretrained("checkpoints/bert-base-uncased")
    encoder_config.encoder_width = mm_hidden_size
    encoder_config.hidden_size = mm_hidden_size
    encoder_config.num_hidden_layers = num_hidden_layers
    encoder_config.num_attention_heads = 8  # original 12, but 1024 % 12 != 0 will raise error
    # insert cross-attention layer every other block
    encoder_config.add_cross_attention = True
    encoder_config.cross_attention_freq = cross_attention_freq
    encoder_config.query_length = num_query_token
    Qformer = BertLMHeadModel(config=encoder_config)
    query_tokens = nn.Parameter(torch.zeros(1, num_query_token, encoder_config.hidden_size))
    
    return Qformer, query_tokens
