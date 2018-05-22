'''
Implements the Transformer attention as a PyTorch layer, and as described by
the Harvard NLP group in this paper: https://arxiv.org/abs/1706.03762.

Note that gradient descent should be done using Adam optimization where the
learning rate first increases for a number of warmup steps before being
annealed proportionally to the inverse square root of the step number. This is
implemented in the `NoamOpt` class in this file.
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

import copy
import math

def clones(module, N):
    '''
    Produces N copies of a module.
    '''
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def subsequent_mask(size):
    '''
    Creates a mask to zero out subsequent positions using a lower triangle matrix
    '''
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k = 1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0
    
def attention(query, key, value, mask = None, dropout = None):
    '''
    Calculates the scaled dot-product attention, defined as:
        Attention(Q, K, V) = softmax( (Q x K^T)/sqrt(d_k) ) x V
    (T denotes a matrix transpose, and x denotes matrix multiplication)

    The input to the attention consists of queries and keys of dimension `d_k`
    and values of dimension `d_v`. We compute the dot products of the query with
    all keys, divide each by `sqrt(d_k)`, and apply a softmax to obtain the
    weights on the values.

    In practice, this is done by computing the attention function on a set of
    queries simultaneously, packed together into a matrix `Q`. The keys and
    values are also packed together into matrices `K` and `V`.
    '''
    d_k = query.size(-1)
    # query:  [nbatches, h, seq_len, d_k]
    # key.T:  [nbatches, h, d_k, seq_len]
    # scores: [nbatches, h, seq_len, seq_len]
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    # matmul(p_attn, value) => [nbatches, h, seq_len, d_k]
    return torch.matmul(p_attn, value), p_attn

    
class TransformerAttention(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model = 512,
                d_ff = 2048, h = 8, N = 6, encode_position = True,
                pos_enc_max_len = 5000, dropout = 0.1):
        '''
        Initializes a Transformer Attention mechanism. The main focus of this
        implementation are the encoder and decoder stacks, but since this type
        of attention mechanism is typically used with positional encoding added
        to the input embeddings, there's the option to do so; this should
        probably only be used when the input to this layer is the output of an
        embedding layer, but maybe there's some innovation I'm missing.
        
            Inputs:
        `src_vocab`: size of the source vocabulary  
        `tgt_vocab`: size of the target vocabulary  
        `d_model`: embedding (and general model) size  
        `d_ff`: size of the position-wise, fully-connected feed-forward layers  
        `h`: number of attention heads  
        `N`: number of encoder and decoder layers  
        `encode_position`: whether or not to add positional encoding to the inputs  
        `pos_enc_max_len`: maximum length of the sequece positions to encode.
        Ignored if `encode_position` is `False`.  
        `dropout`: dropout rate (chance of dropping)  
        '''
        super(TransformerAttention, self).__init__()
        
        # Initialize some basic layers
        self.dropout = nn.Dropout(dropout)
        
        # Initialize a positional encoding layer if needed
        if encode_position:
            self.src_position = PositionalEncoding(d_model, dropout, pos_enc_max_len)
            self.tgt_position = PositionalEncoding(d_model, dropout, pos_enc_max_len)
            
        # Initialize the encoder and decoder layers
        self.encoder_stack = nn.ModuleList([
            EncoderLayer(d_model, d_ff, h, dropout) for _ in range(N)
        ])
        self.decoder_stack = nn.ModuleList([
            DecoderLayer(d_model, d_ff, h, dropout) for _ in range(N)
        ])
        # Encoder and decoder normalizations
        self.enc_norm = LayerNorm(d_model)
        self.dec_norm = LayerNorm(d_model)
        
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.N = N
        self.d_model = d_model
        self.d_ff = d_ff
        self.h = h
        self.encode_position = encode_position
        self.pos_enc_max_len = pos_enc_max_len
        
    def forward(self, src, tgt, src_mask = None, tgt_mask = None):
        '''
        IN: `[batch_size, seq_len, embed_size]`  
        OUT: `[batch_size, seq_len, embed_size]`  
        '''
        if self.encode_position:
            # Apply positional encoding
            src = self.src_position(src)
            tgt = self.tgt_position(tgt)
            
        # Run the input source through the encoder
        memory = src
        for encoder in self.encoder_stack:
            memory = encoder(memory, src_mask)
        memory = self.enc_norm(memory)
            
        # Input the encoder memory to the decoder and run it forward
        output = tgt
        for decoder in self.decoder_stack:
            output = decoder(output, memory, src_mask, tgt_mask)
        output = self.dec_norm(output)
            
        return output

            
class EncoderLayer(nn.Module):
    '''
    Implements a single encoder layer. Each layer has two sublayers:
        1. a multi-headed self-attention mechanism
        2. a position-wise, fully-connected feed-forward network
    The output of each sublayer has a residual connection added, and then is
    passed through a normalization layer before being passed on as the input to
    the next sublayer.
    '''
    def __init__(self, d_model, d_ff, h, dropout):
        super(EncoderLayer, self).__init__()
        # Basic layers
        self.dropout = nn.Dropout(dropout)
        
        # Multi-headed self-attention sublayer
        self.attn = MultiHeadedAttention(h, d_model, dropout)
        # Attention output normalization
        self.attn_norm = LayerNorm(d_model)
        # Position-wise, fully-connected feed-forward sublayer
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        # PositionwiseFeedForward output normalization
        self.ff_norm = LayerNorm(d_model)
        
        self.size = d_model
        self.d_ff = d_ff
        self.h = h
    
    def forward(self, x, mask = None):
        '''
        IN: `[batch_size, seq_len, embed_size]`  
        OUT: `[batch_size, seq_len, embed_size]`  
        '''
        # Normalize input
        x_norm = self.attn_norm(x)
        # Get the output of the multi-headed self-attention mechanism
        attn_out = self.attn(x_norm, x_norm, x_norm, mask)
        # Add the residual connections
        attn = x + self.dropout(attn_out)
        
        # Normalize the residual attention output
        attn_norm = self.attn_norm(attn)
        # Get the output of the position-wise, fully-connected feed-forward network
        ff_out = self.ff(attn_norm)
        # Add the residual connections
        ff = attn_norm + self.dropout(ff_out)
        
        return ff

        
class DecoderLayer(nn.Module):
    '''
    Implements a single decoder layer. Each layer has three sublayers, the first
    and last being the same as in an encoder layer (with the first sublayer's
    inputs masked):
        1. a masked multi-head self-attention mechanism. The masking is used to
        prevent positions from attending to subsequent positions; this, combined
        with the fact that the output embeddings are offset by one position,
        ensures that the predictions for position i can depend only on the known
        outputs at positions less than i
        2. a multi-head source attention mechanism. This is performed over the
        output of the encoder stack
        3. a position-wise fully-connected feed-forward network
    Like with the encoder layers, the output of each sublayer has a residual
    connection added, and then is passed through a normalization layer before
    being passed on as the input to the next sublayer.
    '''
    def __init__(self, d_model, d_ff, h, dropout):
        super(DecoderLayer, self).__init__()
        # Basic layers
        self.dropout = nn.Dropout(dropout)
        
        # Masked multi-headed self-attention sublayer
        self.self_attn = MultiHeadedAttention(h, d_model, dropout)
        # Self-attention normalization
        self.self_attn_norm = LayerNorm(d_model)
        # Multi-headed source attention sublayer
        self.src_attn = MultiHeadedAttention(h, d_model, dropout)
        # Source attention normalization
        self.src_attn_norm = LayerNorm(d_model)
        # Position-wise, fully-connected feed-forward sublayer
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        # PositionwiseFeedForward normalization
        self.ff_norm = LayerNorm(d_model)
        
        self.d_model = d_model
        self.d_ff = d_ff
        self.h = h
        
    def forward(self, x, memory, src_mask = None, tgt_mask = None):
        '''
        IN: `[batch_size, seq_len, embed_size]`  
        OUT: `[batch_size, seq_len, embed_size]`  
        '''
        # Normalize input
        x_norm = self.self_attn_norm(x)
        # Masked self-attention output
        self_attn_out = self.self_attn(x_norm, x_norm, x_norm, tgt_mask)
        # Add the residual connections
        self_attn = x + self.dropout(self_attn_out)
        
        # Normalize input
        self_attn_norm = self.src_attn_norm(self_attn)
        # Source attention output
        src_attn_out = self.src_attn(self_attn_norm, memory, memory, src_mask)
        # Add the residual connections
        src_attn = self_attn + self.dropout(src_attn_out)
        
        # Normalize input
        src_attn_norm = self.ff_norm(src_attn)
        # Position-wise, fully-connected feed-forward output
        ff_out = self.ff(src_attn)
        # Add the residual connections
        ff = src_attn_norm + self.dropout(ff_out)
        
        return ff
    

class MultiHeadedAttention(nn.Module):
    '''
    Multi-head attention allows the model to jointly attend to information from
    different representation subspaces at different positions; with a single
    attention head, averaging inhibits this.
        MultiHead(Q, K, V) = Concat(head_1, ..., head_h) x WO
    where:
        head_i = Attention(Q x WQ_i, K x WK_i, V x WV_i)
    where the projections are parameter matrices (of sizes):
        WQ_i (d_model, d_k), WK_i (d_model, d_k), WV_i (d_model, d_v)
    The paper uses `h=8` parallel attention layers/heads.
    '''
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
        # Each is of size [nbatches, h, seq_len, d_k]
        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]

        # Apply attention on all the projected vectors in the batch
        x, self.attn = attention(query, key, value, mask = mask, dropout = self.dropout)

        # Concat using a view and apply a final linear transformation
        # [nbatches, seq_len, h, d_k]
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h*self.d_k)
        return self.linears[-1](x)

    
class PositionwiseFeedForward(nn.Module):
    '''
    Implements the position-wise fully-connected feed-forward networks used by
    both the encoder and decoder layers. This network consists of two linear
    transforms with a ReLU activation in between, and is applied to each time
    position separately and identically:
        FFN(x) = max(0, X x W_1 + b_1) x W_2 + b_2
    Another way of describing this is as two convolutions with kernel size 1.
    '''
    def __init__(self, d_model, d_ff, dropout = 0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
    
    
class LayerNorm(nn.Module):
    '''
    Implements a normalization layer
    '''
    def __init__(self, features, eps = 1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim = True)
        std = x.std(-1, keepdim = True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

        
class PositionalEncoding(nn.Module):
    '''
    The model utilizes "positional encodings" summed with the input embeddings
    at the bottom of the encoder and decoder stacks to inject information about
    the positions of the tokens in the sequence (since the model contains no
    recurrence or convolution). The paper uses sine and cosine functions of
    different frequencies:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    '''
    def __init__(self, d_model, dropout, max_len = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        
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
        '''
        IN: `[batch_size, seq_len, embed_size]`  
        OUT: `[batch_size, seq_len, embed_size]`  
        '''
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad = False)
        return self.dropout(x)
        
        