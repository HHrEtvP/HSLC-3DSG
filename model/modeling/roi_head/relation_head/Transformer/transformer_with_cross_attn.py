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
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),  # encoder和decoder
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
                 self_posembed=None, cross_posembed=None):
        super().__init__()
        self.self_attn = MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
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
        self.cross_posembed = cross_posembed

        self.size = d_model

    def with_pos_embed(self, tensor, pos_embed: Optional[torch.Tensor]):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, query, key, query_pos, key_pos, padding_mask):
        """
        :param query: B C Pq
        :param key: B C Pk
        :param query_pos: B Pq 3/6
        :param key_pos: B Pk 3/6
        :return:
        """
        # NxCxP to PxNxC
        if self.self_posembed is not None:
            query_pos_embed = self.self_posembed(query_pos).permute(2, 0, 1)
        else:
            query_pos_embed = None
        if self.cross_posembed is not None:
            key_pos_embed = self.cross_posembed(key_pos).permute(2, 0, 1)
        else:
            key_pos_embed = None

        query = query.permute(1, 0, 2)  # N,B,C
        key = key.permute(1, 0, 2)

        q = k = v = self.with_pos_embed(query, query_pos_embed)
        # q:(L,N,E), k,v:(S,N,E)
        query2 = self.self_attn(q, k, value=v, key_padding_mask=padding_mask)[0]
        query = query + self.dropout1(query2)
        query = self.norm1(query)

        query2 = self.multihead_attn(query=self.with_pos_embed(query, query_pos_embed),
                                     key=self.with_pos_embed(key, key_pos_embed),
                                     value=self.with_pos_embed(key, key_pos_embed))[0]
        query = query + self.dropout2(query2)
        query = self.norm2(query)

        query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
        query = query + self.dropout3(query2)
        query = self.norm3(query)

        # NxCxP to PxNxC
        query = query.permute(1, 0, 2)
        torch.cuda.empty_cache()
        return query


class TransformerEncoder(nn.Module):
    # "Core encoder is a stack of N layers"
    def __init__(self, N, d_model, nhead, dim_feedforward, dropout, activation, self_posembed, cross_posembed):
        super(TransformerEncoder, self).__init__()
        self.encoder = nn.ModuleList()
        for i in range(N):
            self.encoder.append(
                TransformerEncoderLayer(
                    d_model, nhead, dim_feedforward, dropout, activation,
                    self_posembed=self_posembed[i],
                    cross_posembed=cross_posembed[i]
                ))

    def forward(self, obj_x, point_x, obj_pos, point_pos, mask):
        query = obj_x
        key = point_x
        query_pos = obj_pos
        key_pos = point_pos
        for encoder in self.encoder:
            query = encoder(query, key, query_pos, key_pos, mask)
        return query

"""
class TransformerStack(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation,
                 self_posembed):
        super().__init__()
        self.transformers = nn.ModuleList()
        for i in range(6): 
            encoder = TransformerEncoder(TransformerEncoderLayer(
                d_model[i],
                nhead[i],
                dim_feedforward[i],
                dropout[i],
                activation[i],
                self_posembed[i]
            ), 6)
            self.transformers.append(encoder)

    def forward(self, x_label, x_visual, pos, mask):
        input = torch.cat([x_label, x_visual], dim=-1)
        input = input + pos
        output = self.transformers[0](input, pos, mask)
        '''
        output = self.transformers[0](x_label, pos, mask)
        output = torch.cat([output, x_visual], dim=-1)
        output = self.transformers[1](output, pos, mask)
        '''
        return output
"""


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

