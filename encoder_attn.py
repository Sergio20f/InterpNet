import torch
import torch.nn as nn

# from def_attn_embedder import AttnEmbedder # TO-DO: build custom
from attn_embedder import AttnEmbedder
#from def_attn_components import get_clones, EncoderLayer # TO-DO: build custom
from attn_utils import get_clones, EncoderLayer


class EncoderTransformer(nn.Module):
    def __init__(self, config):
        super(EncoderTransformer, self).__init__()

        self.N = config["model::N_attention_blocks"]
        self.input_dim = config["model::i_dim"]
        self.embed_dim = config["model::dim_attention_embedding"]
        self.normalize = config["model::normalize"]
        self.heads = config["model::N_attention_heads"]
        self.dropout = config["model::dropout"]
        self.d_ff = config["model::attention_hidden_dim"]
        self.latent_dim = config["model::latent_dim"]
        self.device = config["device"]
        print(f"init attn encoder")

        # Let's encode all of the weights of one neuron together using attn (the code also has the option of
        # encoding each weight separately).
        print("## attention encoder -- use index_dict") # TO-DO: Check what is index_dict
        index_dict = config.get("model::index_dict", None) # TO-DO: Check what is index_dict in their config
        d_embed = config.get("model::attn_embedder_dim")
        n_heads = config.get("model::attn_embedder_nheads")
        self.token_embeddings = AttnEmbedder(i_ndex_dict=index_dict,
                                             d_model=int(self.embed_dim), # TO-DO: Check parameter
                                             d_embed=d_embed,
                                             n_heads=n_heads,
                                             )
        self.max_seq_len = self.token_embeddings.__len__()

        # Original code uses an optional compression token embedding (compress the input sequence before transformer)

        # Get learned position embedding
        self.position_embedding = nn.Embedding(self.max_seq_len, self.embed_dim)

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
        
        # fc to map to latent space
        # bottleneck = config.get("model::bottleneck", "linear") # for now we stick to linear - ignore MLP
        bottleneck_input = self.embed_dim * self.max_seq_len
        
        # if bottleneck == "linear":
        self.vec2neck = nn.Sequential(nn.Linear(bottleneck_input, self.latent_dim), nn.Tanh())

    def forward(self, x, mask=None):
        attn_scores = [] # quote original code: not yet implemented, to prep interface
        # embedd weights
        x = self.token_embeddings(x)
        
        # Adding positional embeddings
        positions = torch.arange(self.max_seq_len, device=x.device).unsqueeze(0)
        x = x + self.position_embeddings(positions).expand_as(x)

        # Passing the sequence through the encoder layers
        for ndx in range(self.N):
            x, scores = self.layers[ndx](x, mask)
            attn_scores.append(scores)

        # compressing the sequence to a fixed-size vector (bottleneck)
        x = x.view(x.shape[0], x.shape[1] * x.shape[3])
        x = self.vec2neck(x)

        return x, attn_scores
