'''
Implements the Transformer attention as a PyTorch layer, and as described by
the Harvard NLP group in this paper: https://arxiv.org/abs/1706.03762
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

import copy
import math
import time

# Helper function that produces N copies of module
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

# Helper function that creates a mask to zero out subsequent positions (using
# a lower triangle matrix)
def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k = 1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

'''
Initializes a Transformer Attention model
'''
class TransformerAttention(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, N = 6, d_model = 512,
                d_ff = 2048, h = 8, dropout = 0.1):
        # Convenience
        c = copy.deepcopy

        attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)
        self.model = EncoderDecoder(
            Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
            Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
            nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
            nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
            Generator(d_model, tgt_vocab)
        )
        
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.N = N
        self.d_model = d_model
        self.d_ff = d_ff
        self.h = h
        self.dropout = dropout

    # Initialize parameters using Glorot/fan_avg
    def init(self):
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform(p)

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.model(src, tgt, src_mask, tgt_mask)

'''
Base container for an encoder/decoder-type model. The actual encoder and
decoder models are passed in at initalization.
'''
class EncoderDecoder(nn.Module):
    '''
    `encoder`  : PyTorch module that takes as input `(src, src_mask)` and
        returns the encoded input
    `decoder`  : PyTorch module that takes as input `(src, src_mask, tgt, tgt_mask)`
        and returns the decoded input
    `src_embed`: PyTorch module that takes as input `src` and returns an
        embedded representation
    `tgt_embed`: PyTorch module that takes as input `src` and returns an
        unembedded representation
    `generator`: PyTorch module with a softmax output
    '''
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    # Step forward using the masked source and target data
    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(
            self.encode(src, src_mask), src_mask, tgt, tgt_mask
        )

    # Step forward, encoding the masked source data
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    # Step forward, decoding the masked target data
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

'''
Standard linear + softmax layer
'''
class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim = -1)

'''
Implements an encoder module as a stack of N layers. The paper uses N=6 layers.
'''
class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm(LayerNorm, layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

'''
Implements a decoder module as a stack of N layers. The paper uses N=6 layers.
'''
class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

'''
Implements a normalization layer
'''
class LayerNorm(nn.Module):
    def __init__(self, features, eps = 1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim = True)
        std = x.std(-1, keepdim = True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

'''
Implements a residual connection followed by a normalization layer
'''
class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    # Feed-forward step into the sublayer, plus a residual connection and layer
    # normalization on the other side
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
'''
Implements a single encoder layer. Each layer has two sublayers:
    1. a multi-head self-attention mechanism
    2. a position-wise fully-connected feed-forward network
The output of each sublayer has a residual connection added, and then is
passed through a normalization layer before being passed on asthe input to the
next sublayer.
'''
class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

'''
Implements a single decoder layer. Each layer has three sublayers, the first
and last being the same as in an encoder layer (with the first sublayer's
inputs masked):
    1. a masked multi-head self-attention mechanism. The masking is used to
       prevent positions from attending to subsequent positions; this, combined
       with the fact that the output embeddings are offset by one position,
       ensures that the predictions for position i can depend only on the known
       outputs at positions less than i
    2. a multi-head self-attention mechanism. This is performed over the output
       of the encoder stack
    3. a position-wise fully-connected feed-forward network
Like with the encoder layers, the output of each sublayer has a residual
connection added, and then is passed through a normalization layer before being
passed on as the input to the next sublayer.
'''
class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

'''
Their Scaled Dot-Product Attention is computed as:

    Attention(Q, K, V) = softmax( (Q x K^T)/sqrt(d_k) ) x V

      (T denotes a matrix transpose, and x denotes matrix multiplication)
The input to the attention consists of queries and keys of dimension d_k and
values of dimension d_v. We compute the dot products of the query with all keys,
divide each by sqrt(d_k), and apply a softmax to obtain the weights on the
values.

In practice, this is done by computing the attention function on a set of
queries simultaneously, packed together into a matrix Q. The keys and values
are also packed together into matrices K and V.
'''
def attention(query, key, value, mask = None, dropout = None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

'''
Multi-head attention allows the model to jointly attend to information from
different representation subspaces at different positions; with a single
attention head, averaging inhibits this.

    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) x WO

where:

    head_i = Attention(Q x WQ_i, K x WK_i, V x WV_i)

where the projections are parameter matrices (of sizes):

    WQ_i (d_model, d_k), WK_i (d_model, d_k), WV_i (d_model, d_v)

The paper uses h=8 parallel attention layers/heads.
'''
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout = 0.1):
        super(MultiHeadedAttention, self).__init__()
        # Ensure number of heads is a multiple of the embedding size
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        # This will be all of our weight matrices: WQ, WK, WV, WO
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p = dropout)

    def forward(self, query, key, value, mask = None):
        if mask is not None:
            # Same mask applied to all h heads
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # Perform batched linear projections: d_model => h x d_k
        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]

        # Apply attention on all the projected vectors in the batch
        x, self.attn = attention(query, key, value, mask = mask, dropout = self.dropout)

        # Concat using a view and apply a final linear transformation
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h*self.d_k)
        return self.linears[-1](x)

'''
Implements the position-wise fully-connected feed-forward networks used by both
the encoder and decoder layers. This network consists of two linear transforms
with a ReLU activation in between, and is applied to each time position
separately and identically:

    FFN(x) = max(0, X x W_1 + b_1) x W_2 + b_2

Another way of describing this is as two convolutions with kernel size 1.
'''
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout = 0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

'''
The paper utilizes tied embeddings (where the encoder and decoder embedding
layers share the same weight matrix). The embeddings are also scaled by sqrt(d_model).
'''
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

'''
The model utilizes "positional encodings" summed with the input embeddings at
the bottom of the encoder and decoder stacks to inject information about the
positions of the tokens in the sequence (since the model contains no recurrence
or convolution). The paper uses sine and cosine functions of different
frequencies:

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
'''
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p = dropout)

        # Compute the positional encodings once in log space
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0)/d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        # `register_buffer()` is used to register buffers that aren't model parameters
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad = False)
        return self.dropout(x)

    