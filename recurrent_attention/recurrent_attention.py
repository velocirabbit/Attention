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
        `alignment`: a PyTorch module to use as the alignment subnetwork. If
            `None`, defaults to a single-layer, fully-connected, linear network.
            The input to this layer should be of size `[..., in_size+h_size]`
            (with `in_size` accounting for whether or not the inputs will be made
            bi-directional), and the output should be of size
            `[..., prealignment_size]`
        `smooth_align`: if true, passes the alignment vector logits through a
            sigmoid function immediately before passing them through a softmax.
            This reduces the effect very large logits have on the computed
            alignment vector and also makes it easier for the model to consider
            multiple locations within the sequence as "important"
        `dropout`: dropout rate (chance of dropping)
        `align_location`: if `True`, implements location-aware alignment by
            taking the previous alignment vector at each step and convolving it
            with a matrix to obtain a location vector
        `loc_align_size`: the number of vectors to extract for each position/element
            in the previous alignment vector (i.e. the number of output channels 
            in the convolution kernel)
        `loc_align_kernel`: size of the location alignment kernel used to obtain
            the location alignment vector (i.e. the number of positions in the
            sequence the kernel looks at during each convolution step). Padding
            is automatically calculated and added to make sure the convolution
            output has the same sequence length as the input. This should be an
            odd integer (and even integer inputs will be increased by 1) since
            the amount of padding used is `(loc_align_kernel - 1)/2`
        `attention`: a PyTorch module to use as the attention subnetwork. If
            `None`, defaults to a single-layer, fully-connected, linear network.
            The input to this layer should be of size `[..., in_size+h_size+out_size]`,
            and the output should be logits rather than softmax probabilities
            and of size `[..., out_size]`
        `attn_act_fn`: type of activation function to apply to the output of the
            attention subnetwork; after activation, this will be fed back into
            recurrent subnetwork. If `None`, a Linear activation is used
        `rnn_type`: type of RNN cell to use. Should either be `'lstm'` or `'gru'`
            because that's all I've planned for :/
        `num_rnn_layers`: number of recurrent layers in the recurrent subnetwork
        `bidirectional`: whether or not the input sequence to the recurrent
            subnetwork should be made bi-directional (if `True`, `in_size` will
            be doubled)
    '''
    def __init__(self, in_size, h_size, out_size, 
                alignment = None, smooth_align = False, dropout = 0.1,
                align_location = False, loc_align_size = 1, loc_align_kernel = 1,
                attention = None, attn_act_fn = 'Softmax', rnn_type = 'lstm',
                num_rnn_layers = 1, bidirectional = False):
        super(RecurrentAttention, self).__init__()
        # Fix input parameters
        rnn_type = rnn_type.upper()
        in_size *= 2 if bidirectional else 1
        num_rnn_states = 2 if rnn_type == 'LSTM' else 1

        # Initialize some basic layers
        self.drop = nn.Dropout(dropout)
        self.softmax = nn.Softmax(0)
        self.sigmoid = nn.Sigmoid()

        ### Initialize subnetworks ###
        # Recurrent subnetwork
        rnn = getattr(nn, rnn_type+'Cell')
        self.rnn_stack = nn.ModuleList([
            rnn(in_size + out_size if i == 0 else h_size, h_size)
            for i in range(num_rnn_layers)
        ])

        # Alignment subnetwork
        if align_location:
            # Calculate the amount of padding that needs to be applied to each
            # tensor passed in as an input to the location alignment convolution.
            # The amount of padding added is such that inputs and outputs have
            # the same sequence length regardless of the kernel size used, and
            # is given by: (loc_align_kernel - 1)/2
            if loc_align_kernel % 2 == 0:  # make sure loc_align_kernel is odd
                loc_align_kernel += 1
            pad = (loc_align_kernel - 1) // 2
            self.loc_align = nn.Conv1d(
                1, loc_align_size, loc_align_kernel, padding = pad
            )
        if alignment is None:
            align_in_size = in_size + h_size + (loc_align_size if align_location else 0)
            alignment = nn.Linear(align_in_size, 1)
        self.alignment = alignment

        # Attention subnetwork
        if attention is None:
            attention = nn.Linear(in_size + h_size + out_size, out_size)
        self.attention = attention
        
        # Attention output activation (for recurrent inputs)
        if attn_act_fn is not None:
            fn_type = attn_act_fn
            if fn_type in ['Softmax', 'LogSoftmax']:
                attn_act_fn = getattr(nn, fn_type)(-1)
            else:
                attn_act_fn = getattr(nn, fn_type)()
        self.attn_act_fn = attn_act_fn
        ################################################
        # For visualization
        self.save_attn_wts = False
        self.attn_wts = None
        self.rnn_states = None

        # Save parameters
        self.in_size = in_size
        self.h_size = h_size
        self.out_size = out_size
        self.align_location = align_location
        self.loc_align_size = loc_align_size
        self.loc_align_kernel = loc_align_kernel
        self.smooth_align = smooth_align
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
        y = Variable(torch.zeros(batch_size, self.out_size))
        c = Variable(torch.zeros(batch_size, self.in_size))
        return (y, c, rnn_states)

    # Run a batch of inputs forward through the recurrent attention mechanism
    # input : [seq_len, batch_size, in_size]
    # output: [seq_len, batch_size, out_size]
    def forward(self, inputs, states):
        # Get the recurrent states and the initial output vector
        y, c, rnn_states = states
        seq_len = inputs.size(0)
        batch_size = inputs.size(1)

        # Create bi-directional inputs
        if self.bidirectional:
            # Get a flipped version of the input vector sequence
            rev_idx = range(seq_len-1, 0, -1)
            flipped = torch.index_select(inputs, 0, rev_idx)
            inputs = torch.cat([inputs, flipped], dim = -1)                     # [seq_len, batch_size, in_size]
        
        # If we're going to be aligning the location, need to initialize an
        # alignment vector of size [seq_len, batch_size, 1] to use
        if self.align_location:
            align_wts = Variable(torch.zeros(seq_len, batch_size, 1))
        y_seq = []
        # For visualization purposes
        if self.save_attn_wts:
            all_attn_wts = []
            all_rnn_states = []
        # Pass each step of the input one at a time
        for _ in range(seq_len):
            # Concatenate y and c to get the input to the RNN this step
            rnn_in = torch.cat([self.drop(c), y], -1)                           # [batch_size, in_size+out_size]
            # Run this through the recurrent subnetwork
            new_rnn_states = []
            for rnn, rnn_state in zip(self.rnn_stack, rnn_states):
                new_states = rnn(rnn_in, rnn_state)  # new_states := (new_h, new_c)
                new_rnn_states.append(new_states)
                rnn_in = new_states if not isinstance(new_states, tuple) else new_states[0]
            if self.save_attn_wts:
                all_rnn_states.append(new_rnn_states)
            # The hidden state of the last cell is the input to the alignment
            s = new_states if not isinstance(new_states, tuple) else new_states[0]
            # Get the alignment weights
            _s_ = (self.drop(s)).expand(seq_len, -1, -1)                        # [seq_len, batch_size, h_size]
            align_in = torch.cat([inputs, _s_], -1)                             # [seq_len, batch_size, in_size+h_size]
            # Convolve location alignment kernel over previous alignment vector
            if self.align_location:
                # Reshape align_wts to match what Conv1d requires
                _align_wts_ = align_wts.permute(1, 2, 0)                        # [batch_size, 1, seq_len]
                _loc_align_ = self.loc_align(_align_wts_)                       # [batch_size, loc_align_size, seq_len]
                loc_align = _loc_align_.permute(2, 0, 1)                        # [seq_len, batch_size, loc_align_size]
                align_in = torch.cat([align_in, loc_align], -1)                 # [seq_len, batch_size, in_size+h_size+loc_align_size]
            align_logits = self.alignment(align_in)                             # [seq_len, batch_size, 1]
            # Logit smoothing
            if self.smooth_align:
                align_logits = self.sigmoid(align_logits)
            align_wts = self.softmax(align_logits)                              # [seq_len, batch_size, 1]
            if self.save_attn_wts:
                all_attn_wts.append(align_wts.data.clone())
            # Get the context vector for this step as a weighted sum of all steps
            c = inputs.mul(align_wts).sum(0)  # `sum()` squeeze by default      # [batch_size, in_size]
            # Get the output vector for this step
            attn_in = self.drop(torch.cat([y, s, c], -1))                       # [batch_size, out_size+h_size+in_size]
            y = self.attention(attn_in)                                         # [batch_size, out_size]
            # Apply an activation function to the attention output
            if self.attn_act_fn is not None:
                y = self.attn_act_fn(y)
            # Append this output to the output sequence
            y_seq.append(y.unsqueeze(0))                                        # [1, batch_size, out_size]
        # Concatenate the output sequence into a single tensor
        outputs = torch.cat(y_seq, 0)                                           # [seq_len, batch_size, out_size]
        # Concatenate all of the thing we need for visualization
        if self.save_attn_wts:
            self.attn_wts = torch.cat(all_attn_wts, -1)
            if isinstance(rnn_states[0], tuple):
              self.rnn_states = [
                  (
                      torch.cat([  # Hidden states
                          all_rnn_states[t][i][0].unsqueeze(-1) for t in range(seq_len)
                      ], -1).data.clone(),
                      torch.cat([  # Cell states
                          all_rnn_states[t][i][1].unsqueeze(-1) for t in range(seq_len)
                      ], -1).data.clone()
                  ) for i in range(self.num_rnn_layers)
              ]
            else:
              self.rnn_states = [
                  torch.cat([
                      all_rnn_states[t][i].unsqueeze(-1) for t in range(seq_len)
                  ], -1).data.clone() for i in range(self.num_rnn_layers)
              ]
        # Return the output sequence, last output vector y, and the new RNN states
        return outputs, (y, c, new_rnn_states)
