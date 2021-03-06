import numpy as np
import torch
import torch.nn as nn

from torch.autograd import Variable
from recurrent_attention import RecurrentAttention

class RNNModel(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, embed_size, h_size, align_size,
                 decode_size, n_enc_layers, attn_rnn_layers, n_dec_layers,
                 encode_size = 0, decode_out_size = 0, align_location = False,
                 loc_align_size = 1, loc_align_kernel = 1, smooth_align = False,
                 tie_wts = True, bidirectional_attn = False,
                 skip_connections = False, dropout = 0.1):
        super(RNNModel, self).__init__()
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.embed_size = embed_size
        self.encode_size = encode_size
        self.h_size = h_size
        self.align_size = align_size
        self.decode_size = decode_size
        self.decode_out_size = decode_out_size
        self.n_enc_layers = n_enc_layers
        self.attn_rnn_layers = attn_rnn_layers
        self.n_dec_layers = n_dec_layers
        self.align_location = align_location
        self.loc_align_size = loc_align_size
        self.loc_align_kernel = loc_align_kernel
        self.smooth_align = smooth_align
        self.tie_wts = tie_wts
        self.bidirectional_attn = bidirectional_attn
        self.skip_connections = skip_connections
        self.dropout = dropout
        
        # Basic layers
        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
        # Embedding layers
        self.embedding = nn.Embedding(src_vocab, embed_size)
        
        # Encoder layers
        if n_enc_layers > 0:
            self.encoders = nn.ModuleList([
                nn.LSTM(
                    input_size = embed_size if i == 0 else encode_size,
                    hidden_size = encode_size, dropout = dropout
                ) for i in range(n_enc_layers)
            ])
            attn_in_size = encode_size
        else:
            self.encoders = None
            attn_in_size = embed_size

        # Recurrent attention mechanism
        self.attn = RecurrentAttention(
            in_size = attn_in_size, h_size = h_size, align_size = align_size,
            out_size = decode_size, align_location = align_location,
            loc_align_size = loc_align_size, loc_align_kernel = loc_align_kernel,
            smooth_align = smooth_align, num_rnn_layers = attn_rnn_layers,
            attn_act_fn = 'ReLU', dropout = dropout, bidirectional = bidirectional_attn
        )

        # Decoder layers
        if n_dec_layers > 0:
            self.decoders = nn.ModuleList([
                nn.LSTM(
                    input_size = decode_size, dropout = dropout,
                    hidden_size = decode_size if i < n_dec_layers-1 else decode_out_size,
                ) for i in range(n_dec_layers)
            ])
            project_in_size = decode_out_size
        else:
            self.decoders = None
            project_in_size = decode_size

        if skip_connections:
            # Create a linear layer to transform the embedding output to be of
            # encode_size so that it can be used in the skip connections
            if n_enc_layers > 0:
                self.embed_skip = nn.Linear(embed_size, encode_size)
            # Do the same for the output of the second-to-last decoder layer to
            # the output of the last decoder layer
            if n_dec_layers > 0:
                self.decode_skip = nn.Linear(decode_size, decode_out_size)

        # Projection layer
        self.projection = nn.Linear(project_in_size, tgt_vocab, bias = False)

        # Tie weights
        if tie_wts and src_vocab == tgt_vocab and embed_size == project_in_size:
            self.embedding.weight = self.projection.weight
        
        # For output
        self.log_softmax = nn.LogSoftmax(dim = -1)
            
        # For visualizations
        self.save_wts = False
        self.enc_out = None
        self.dec_out = None

        # Initialize
        self.init()
        
    def init(self):
        recurrent_layers = []
        if self.n_enc_layers > 0:
            recurrent_layers.append(self.encoders)
        if self.n_dec_layers > 0:
            recurrent_layers.append(self.decoders)
        # Recurrent layers get orthogonal initialization
        for subnet in recurrent_layers:
            for layer in subnet:
                for p in layer.parameters():
                    if p.dim() > 1:
                        nn.init.orthogonal(p)
                    else:
                        p.data.fill_(0)
        for p in self.projection.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform(p)
            else:
                p.data.fill_(0)
        self.attn.init()
        
    def init_states(self, batch_size):
        pkg = []
        if self.n_enc_layers > 0:
            encoder_states = [
                (
                    Variable(torch.zeros(1, batch_size, self.encode_size)),
                    Variable(torch.zeros(1, batch_size, self.encode_size))
                ) for _ in range(self.n_enc_layers)
            ]
            pkg.append(encoder_states)
        
        attn_states = self.attn.init_rnn_states(batch_size)
        pkg.append(attn_states)

        if self.n_dec_layers > 0:
            decoder_states = [
                (
                    Variable(torch.zeros(
                        1, batch_size, self.decode_size if i < self.n_dec_layers-1 else self.decode_out_size
                    )),
                    Variable(torch.zeros(
                        1, batch_size, self.decode_size if i < self.n_dec_layers-1 else self.decode_out_size
                    ))
                ) for i in range(self.n_dec_layers)
            ]
            pkg.append(decoder_states)

        return tuple(pkg)
    
    def forward(self, inputs, states):
        if self.n_enc_layers > 0:
            enc_states = states[0]
            states = states[1:]
        if self.n_dec_layers > 0:
            dec_states = states[-1]
            states = states[:-1]
        attn_states = states[0]
        pkg = []
        if self.save_wts:
            self.enc_out = []
            self.dec_out = []
        
        # Embedding layer
        embeddings = self.embedding(inputs) * np.sqrt(self.embed_size)
        
        # Encoder stack
        if self.n_enc_layers > 0:
            new_enc_states = []
            enc_in = self.drop(self.relu(embeddings))
            for i, (states, encoder) in enumerate(zip(enc_states, self.encoders)):
                enc_out, new_enc_state = encoder(enc_in, states)
                new_enc_states.append(new_enc_state)
                if self.save_wts:
                    self.enc_out.append(enc_out.data.clone())
                if self.skip_connections:
                    if i == 0:
                        enc_in = self.drop(self.embed_skip(enc_in))
                    enc_out = enc_out + enc_in
                enc_in = self.drop(enc_out)
            attn_in = enc_in
            pkg.append(new_enc_states)
        else:
            attn_in = embeddings
                
        # Attention mechanism
        attn_out, new_attn_states = self.attn(attn_in, attn_states)
        pkg.append(new_attn_states)
        
        # Decoder stack
        if self.n_dec_layers > 0:
            new_dec_states = []
            dec_in = attn_out
            for i, (states, decoder) in enumerate(zip(dec_states, self.decoders)):
                dec_out, new_dec_state = decoder(dec_in, states)
                new_dec_states.append(new_dec_state)
                if self.save_wts:
                    self.dec_out.append(dec_out.data.clone())
                if self.skip_connections:
                    if i == self.n_dec_layers - 1:
                        dec_in = self.drop(self.decode_skip(dec_in))
                    dec_out = dec_out + dec_in
                dec_in = self.drop(dec_out)
            proj_in = dec_in
            pkg.append(new_dec_states)
        else:
            proj_in = attn_out
        
        # Projection layer
        logits = self.projection(proj_in)
        output = self.log_softmax(logits)
        
        return output, tuple(pkg)
    
    def train(self, mode = True, save_wts = False):
        super(RNNModel, self).train(mode)
        self.attn.save_attn_wts = save_wts
        self.save_wts = save_wts
        
    def eval(self, save_wts = True):
        super(RNNModel, self).eval()
        self.attn.save_attn_wts = save_wts
        self.save_wts = save_wts


class LabelSmoothing(nn.Module):
    def __init__(self, size, padding_idx = None, smoothing = 0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss()
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = torch.zeros_like(x.data)
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist.add_(self.smoothing / self.size)
        if self.padding_idx is not None:
            true_dist[:, self.padding_idx] = 0
            mask = torch.nonzero(target.data == self.padding_idx)
            if mask.dim() > 0:
                true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        loss = self.criterion(x, Variable(true_dist, requires_grad = False))
        return loss * self.size