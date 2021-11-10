import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from.multi_head_attention import MultiheadAttention


'''
class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                           tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


def make_model(src_vocab, tgt_vocab, N=6,
               d_model=512, d_ff=2048, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N), 
        Decoder(DecoderLayer(d_model, c(attn), c(attn),
                             c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)), 
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
    )
    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model
'''


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead=8, dim_feedforward=2048, dropout=0.1, activation="relu",
                 self_posembed=None):
        super().__init__()
        self.self_attn = MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout)
        # self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)


        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        self.self_posembed = self_posembed

        self.size = d_model

    def with_pos_embed(self, tensor, pos_embed: Optional[torch.Tensor]):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, query, query_pos, padding_mask):
        if self.self_posembed is not None:
            query_pos_embed = self.self_posembed(query_pos).permute(2, 0, 1)
        else:
            query_pos_embed = None
        """
        if self.cross_posembed is not None:
            key_pos_embed = self.cross_posembed(key_pos).permute(2, 0, 1)
        else:
            key_pos_embed = None
        """

        query = query.permute(1, 0, 2)  # (N,L,E)->(L,N,E)
        # key = key.permute(2, 0, 1)

        q = k = v = self.with_pos_embed(query, query_pos_embed)
        # q:(L,N,E), k,v:(S,N,E)
        query2 = self.self_attn(query=q, key=k, value=v, key_padding_mask=padding_mask)[0]
        query = query + self.dropout1(query2)
        query = self.norm1(query)
        """
        query2 = self.multihead_attn(query=self.with_pos_embed(query, query_pos_embed),
                                     key=self.with_pos_embed(key, key_pos_embed),
                                     value=self.with_pos_embed(key, key_pos_embed))[0]
        query = query + self.dropout2(query2)
        query = self.norm2(query)
        """
        '''FFN'''
        query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
        query = query + self.dropout3(query2)
        query = self.norm3(query)
        query = query.permute(1, 0, 2)
        return query


class TransformerEncoder(nn.Module):
    # "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(TransformerEncoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x, x_pos, mask):
        for layer in self.layers:
            x = layer(x, x_pos, mask)
        return self.norm(x)


class TransformerStack(nn.Module):
    def __init__(self, N, d_model, nhead, dim_feedforward, dropout, activation,
                 self_posembed):
        super().__init__()
        self.transformers = nn.ModuleList()
        self.num_decoder_layers = N

        self.decoder = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            self.decoder.append(
                TransformerEncoderLayer(
                    d_model, nhead, dim_feedforward, dropout, activation,
                    self_posembed=self_posembed[i],
                ))

    def forward(self, x, pos, mask):
        for i in range(self.num_decoder_layers):
            # Transformer Decoder Layer
            x = self.decoder[i](x, pos, mask)
        return x


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

