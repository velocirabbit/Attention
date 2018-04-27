'''
Recurrent Attention model, as described in "Neural Machine Translation by
Jointly Learning to Align and Translate" (Bahdanau, Cho, Bengio).
'''
import numpy as np
import torch
import torch.nn as nn

from torch.autograd import Variable
from torch.nn import functional as F

class RecurrentAttention(nn.Module):
    '''
    Initializes a recurrent attention mechanism.
    Inputs:
        `in_size`: size of the input vectors. If bi-directional, use just the
            size of the vector in just the forward direction, but make sure to
            account for the doubled size if you're passing in your own
            `alignment` layer (the default will handle that automatically)
        `h_size`: size of the recurrent hidden states
        `out_size`: size of the output vectors
        `dropout`: dropout percentage for the input to the recurrent subnetwork
        `alignment`: a PyTorch module to use as the alignment subnetwork. If
            `None`, defaults to a single-layer, fully-connected, linear network.
            The input to this layer should be of size `[..., in_size+h_size]`
            (with `in_size` accounting for whether or not the inputs will be
            made bi-directional), and the output should be logits rather than
            softmax probabilities and of size `[..., 1]`
        `attention`: a PyTorch module to use as the attention subnetwork. If
            `None`, defaults to a single-layer, fully-connected, linear network.
            The input to this layer should be of size `[..., in_size+h_size+out_size]`,
            and the output should be logits rather than softmax probabilities
            and of size `[..., out_size]`
        `rnn_type`: type of RNN cell to use. Should either be `'lstm'` or `'gru'`
            because that's all I've planned for :/
        `rnn_layers`: number of recurrent layers in the recurrent subnetwork
        `bidirectional`: whether or not the input sequence to the recurrent
            subnetwork should be made bi-directional (if `True`, `in_size` will
            be doubled)
    '''
    def __init__(self, in_size, h_size, out_size, dropout = 0.1,
                alignment = None, attention = None, rnn_type = 'lstm',
                num_rnn_layers = 1, bidirectional = False):
        super(RecurrentAttention, self).__init__()
        # Fix input parameters
        rnn_type = rnn_type.upper()
        in_size *= 2 if bidirectional else 1
        num_rnn_states = 2 if rnn_type == 'LSTM' else 1

        # Initialize some basic layers
        self.drop = nn.Dropout(dropout)
        self.softmax = nn.Softmax(0)

        ### Initialize subnetworks ###
        # Alignment subnetwork
        if alignment is None:
            alignment = nn.Linear(in_size + h_size, 1)
        self.alignment = alignment

        # Recurrent subnetwork
        rnn = getattr(nn, rnn_type+'Cell')
        self.rnn_stack = [rnn(in_size + out_size, h_size)]
        for _ in range(1, num_rnn_layers):
            self.rnn_stack.append(rnn(h_size, h_size))

        # Attention subnetwork
        if attention is None:
            attention = nn.Linear(in_size + h_size + out_size, out_size)
        self.attention = attention
        ################################################

        # Save parameters
        self.in_size = in_size
        self.h_size = h_size
        self.out_size = out_size
        self.rnn_type = rnn_type
        self.num_rnn_states = num_rnn_states
        self.num_rnn_layers = num_rnn_layers
        self.bidirectional = bidirectional
        self.dropout = dropout

    # Initialize model parameters
    def init(self):
        # Initialize the alignment and attention weights/biases
        for subnet in [self.alignment, self.attention]:
            for p in subnet.parameters():
                if p.dim() > 1:     # Glorot/Xavier uniform (fan average) init
                    nn.init.xavier_uniform(p)
                else:               # Fill bias vectors with 0's
                    p.data.fill_(0)
        # Initialize the recurrent subnetwork weights/biases
        for layer in self.rnn_stack:
            for p in layer.parameters():
                if p.dim() > 1:     # Glorot/Xavier normal init
                    nn.init.xavier_normal(p)
                else:               # Fill bias vectors with 0's
                    p.data.fill_(0)

    # Initialize and return a set of hidden states for the recurrent network
    # Also initializes a blank attention output vector to use as an initial output
    def init_rnn_states(self, batch_size):
        n_states = self.num_rnn_states
        n_layers = self.num_rnn_layers
        h_size = self.h_size

        # Initialize recurrent states
        rnn_states = [
            tuple([
                Variable(
                    torch.zeros(batch_size, h_size)
                ) for _ in range(n_states)
            ]) for _ in range(n_layers)
        ]
        rnn_states = [s if len(s) > 1 else s[0] for s in rnn_states]
        # Output vector
        output = Variable(torch.zeros(batch_size, self.out_size))
        return (output, rnn_states)

    # Run a batch of inputs forward through the recurrent attention mechanism
    # input : [seq_len, batch_size, in_size]
    # output: [seq_len, batch_size, out_size]
    def forward(self, inputs, states):
        # Get the recurrent states and the initial output vector
        y, rnn_states = states
        seq_len = inputs.size(0)

        # Create bi-directional inputs
        if self.bidirectional:
            # Get a flipped version of the input vector sequence
            rev_idx = range(seq_len-1, 0, -1)
            flipped = torch.index_select(inputs, 0, rev_idx)
            inputs = torch.cat([inputs, flipped], dim = -1)                     # [seq_len, batch_size, in_size]
        
        y_seq = []
        # Pass each step of the input one at a time
        for _ in range(seq_len):
            # Get the hidden state of the first RNN layer. size: [batch_size, h_size]
            s = rnn_states[0] if not isinstance(rnn_states[0], tuple) else rnn_states[0][0]
            # Get the alignment weights
            s = (self.drop(s)).expand(seq_len, -1, -1)                          # [seq_len, batch_size, h_size]
            align_in = torch.cat([inputs, s], -1)                               # [seq_len, batch_size, in_size+h_size]
            align_logs = self.alignment(align_in)                               # [seq_len, batch_size, 1]
            align_wts = self.softmax(align_logs)
            # Get the context vector for this step as a weighted sum of all steps
            align_wts = align_wts.expand_as(inputs)                             # [seq_len, batch_size, in_size]
            c = align_wts.mul(inputs).sum(0).squeeze(0)                         # [batch_size, in_size]
            # Pass the context vector through the recurrent subnetwork
            rnn_in = torch.cat([self.drop(c), y], -1)                           # [batch_size, in_size+out_size]
            new_rnn_states = []
            for rnn, rnn_state in zip(self.rnn_stack, rnn_states):
                new_states = rnn(rnn_in, rnn_state)  # new_states := (new_h, new_c)
                new_rnn_states.append(new_states)
                # The hidden state of this cell is the input to the next layer
                rnn_in = self.drop(new_states[0])                               # [batch_size, h_size]
            # Get the output vector for this step
            attn_in = torch.cat([y, rnn_in, c], -1)                             # [batch_size, in_size+h_size+out_size]
            y = self.attention(attn_in)                                         # [batch_size, out_size]
            # Append this output to the output sequence
            y_seq.append(y.expand(1, -1, -1))
        # Concatenate the output sequence into a single tensor
        outputs = torch.cat(y_seq, 0)
        # Return the output sequence, last output vector y, and the new RNN states
        return outputs, (y, new_rnn_states)



            




                

