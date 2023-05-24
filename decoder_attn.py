import torch
import torch.nn as nn

from attn_utils import get_clones, EncoderLayer, DebedderNeuronGroup_index


class DecoderTransformer(nn.Module):
    def __init__(self, config):
        super(DecoderTransformer, self).__init__()

        # def __init__(self, input_dim, embed_dim, N, heads, max_seq_len, dropout, d_ff):
        self.N = config["model::N_attention_blocks"]
        self.input_dim = config["model::i_dim"]
        self.embed_dim = config["model::dim_attention_embedding"]
        self.normalize = config["model::normalize"]
        self.heads = config["model::N_attention_heads"]
        self.dropout = config["model::dropout"]
        self.d_ff = config["model::attention_hidden_dim"]
        self.latent_dim = config["model::latent_dim"]
        self.device = config["device"]

        index_dict = config.get("model::index_dict", None)
        self.token_embeddings = DebedderNeuronGroup_index(index_dict=index_dict,
                                                          d_model=self.embed_dim)
        self.max_seq_len = self.token_embeddings.__len__()

        # Get learned position embedding
        self.position_embeddings = nn.Embedding(self.max_seq_len, self.embed_dim)

        self.layers = get_clones(
            EncoderLayer(
                d_model=self.embed_dim,
                heads=self.heads,
                normalize=self.normalize,
                dropout=self.dropout,
                d_ff=self.d_ff,
            ),
            self.N,
        )

        # assume, for now, bottleneck is a linear layer
        self.nec2vec = nn.Linear(self.latent_dim, self.embed_dim * self.max_seq_len)

    def forward(self, z, mask=None):
        attn_scores = [] # quote original code: not yet implemented, to prep interface

        # Decompress
        y = self.nec2vec(z)
        y = y.view(z.shape[0], self.max_seq_len, self.embed_dim)

        # Embed positions
        positions = torch.arange(self.max_seq_len, device=y.device).unsqueeze(0)
        y = y + self.position_embeddings(positions).expand_as(y)

        for ndx in range(self.N):
            y, scores = self.layers[ndx](y, mask)
            attn_scores.append(scores)

        # map back to original space.
        y = self.token_debeddings(y)
        
        return y, attn_scores
    