'''
Attention network for seq2seq models.
'''
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import RNN

# Attention network: input of size (batch, time length, state size)
alignment_input = Lambda(lambda x:
    K.concatenate([K.zeros_like(x[:, 0, :]), x[:, 1:, :]], axis = 1)
)(stack_output)

class AttentionCell(Layer):
    """
    Input shape: [return_sequence_shape, state_shape, ...]
        where:
            return_sequence_shape: (batch_size, timesteps, units)
            state_shape, ...:      (batch_size, units) x timesteps
    """
    def __init__(self,
                **kwargs):
        super(AttentionCell, self).__init__(**kwargs)
        

    def build(self, input_shape):
        super(AttentionCell, self).build(input_shape)

    def call(self, x):
        return x
    
    def comput_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        
        return input_shape

class Attention(RNN):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Attention, self).build(input_shape)

    def call(self, x):
        return x

    # input_shape: (batch, time length, state size)
    def compute_output_shape(self, input_shape):
        return input_shape
    




