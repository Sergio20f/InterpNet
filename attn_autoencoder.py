import torch.nn as nn

from encoder_attn import EncoderTransformer
from decoder_attn import DecoderTransformer


class AE_attn(nn.module):
    def __init__(self, config):
        super(AE_attn, self).__init__()

        self.encoder = EncoderTransformer(config)
        self.decoder = DecoderTransformer(config)

    def forward(self, x):
        z = self.forward_encoder(x)
        y = self.forward_decoder(z)
        return z, y

    def forward_encoder(self, x):
        z, _ = self.encoder(x)
        return z

    def forward_decoder(self, z):
        y, _ = self.decoder(z)
        return y
