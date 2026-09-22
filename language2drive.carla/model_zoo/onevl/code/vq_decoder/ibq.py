"""
IBQ (Index-Based Quantization) model.
Adapted from Emu3.5 (BAAI) vision_tokenizer/ibq.py.
"""

from torch import nn

from .modules.encoder_decoder import Encoder, Decoder
from .modules.quantize import IndexPropagationQuantize


class IBQ(nn.Module):

    def __init__(self, ddconfig, n_embed, embed_dim, beta=0.25,
                 use_entropy_loss=False, cosine_similarity=False,
                 entropy_temperature=0.01, sample_minimization_weight=1.0,
                 batch_maximization_weight=1.0, **kwargs):
        super().__init__()
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.quantize = IndexPropagationQuantize(
            n_embed, embed_dim, beta, use_entropy_loss,
            cosine_similarity=cosine_similarity,
            entropy_temperature=entropy_temperature,
            sample_minimization_weight=sample_minimization_weight,
            batch_maximization_weight=batch_maximization_weight,
        )
        self.quant_conv = nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant, return_intermediate_feature=False):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant, return_intermediate_feature=return_intermediate_feature)
        return dec

    def decode_code(self, code_b, shape=None):
        quant_b = self.quantize.get_codebook_entry(code_b, shape=shape)
        dec = self.decode(quant_b)
        return dec
