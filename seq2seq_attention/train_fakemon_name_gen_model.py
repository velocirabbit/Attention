'''
[MPCS 53001] Databases
Andrew Wang

Trains a character-level neural net on the names of all real (in the sense that
they're officially recognized Pokemon created by Nintendo, and, unfortunately,
not in the sense that they exist in our real-life world. Please correct me if
that is not accurate. Please.) Pokemon and a number of the most-common words in
the English language (excluding articles and typical stop words).

Model is basically a 3-stack bi-LSTM that tries to predict the next character
given an input sequence of characters.
'''
from keras.models import Model
from keras.layers import Bidirectional, Dense, Dropout, Embedding, Input, LSTM, LSTMCell, RNN, TimeDistributed
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
import numpy as np
import os
import pickle

USE_BIDIRECTIONAL = False  # Better model, slower to train

PKMN_NAMES_FILE = os.path.join('data', 'real_pkmn_names.txt')
FREQ_WORDS_FILE = os.path.join('data', 'common_english_words.txt')
DICT_PKL = 'c2i_i2c_dict%s.pkl' % ('_bi' if USE_BIDIRECTIONAL else '')
MODEL_NAME = 'fakemon_name_%sgen_model' % ('bi_' if USE_BIDIRECTIONAL else '')
SAVE_MODEL_FILE = os.path.join('checkpoints', MODEL_NAME)

STOP_WORDS = open(os.path.join('data', 'stop_words.txt'), 'r').read().split()
MAX_WORD_LEN = 25          # Longest word allowed, including an SOF or EOF char
STATE_DIM = 180
STACK_SIZE = 3             # Number of Bi-LSTMs in the stack
# Don't train too much! Some noise error makes for more interesting generated names
EPOCHS = (2 if USE_BIDIRECTIONAL else 1) * 50
BATCH_SIZE = 1

USE_PADDING = False        # Only necessary if BATCH_SIZE > 1
INPUT_SHAPE = MAX_WORD_LEN if USE_PADDING else None

SOF_CHAR = '#'             # Signifies the start of the name
EOF_CHAR = '!'             # Signifies the end of the name

'''
Preprocessing
'''
# Load in the training data and do some simple pre-processing
data_pkmn_names = [
    w.lower() for w in open(PKMN_NAMES_FILE, 'r').read().strip().split('\n')
]
data_freq_words = [
    w for w in open(FREQ_WORDS_FILE, 'r').read().split() if w not in STOP_WORDS
]
# Remove 'Porygon2' from data_pkmn_names just because
data_pkmn_names.remove('porygon2')

# Combine the data and remove duplicates, on the very, very, very, very, very,
# very, very, very, very, very, very, very, very, very, very, very, very, very,
# very off chance that a real, officially-recognized Pokemon has a name that's
# also an actual English word.
data = list(set(data_pkmn_names + data_freq_words))

# Get all unique characters in the data
chars = list(set(''.join(data)))
num_chars = 2 + len(chars)  # +2 for `Start of word` and `End of word` chars
# Create a char2index and index2char dictionary
if USE_PADDING:
    num_chars += 1
    c2i = {c:(i+1) for i, c in enumerate(chars)}
    i2c = {(i+1):c for i, c in enumerate(chars)}
else:
    
    c2i = {c:i for i, c in enumerate(chars)}
    i2c = {i:c for i, c in enumerate(chars)}
# Add the SOF and EOF to the dictionaries
SOF_INDEX = num_chars - 2
EOF_INDEX = num_chars - 1
c2i[SOF_CHAR] = SOF_INDEX
c2i[EOF_CHAR] = EOF_INDEX
i2c[SOF_INDEX] = SOF_CHAR
i2c[EOF_INDEX] = EOF_CHAR

# Shuffle for good measure
np.random.shuffle(data)
# Validation set. Technically shouldn't have data in the training set included
# in the validation set, but wtvr, this isn't a production model
val_data = data_pkmn_names + list(np.random.choice(
        data_freq_words, size = len(data_pkmn_names), replace = False
    ))
# Transform the data into sequences of characters (with EOF_INDEX appended)
data = [
    [[SOF_INDEX]] + [[c2i[c]] for c in w] + [[EOF_INDEX]] for w in data
]
val_data = [
    [[SOF_INDEX]] + [[c2i[c]] for c in w] + [[EOF_INDEX]] for w in val_data
]
if USE_PADDING:
    data = pad_sequences(data, maxlen = MAX_WORD_LEN)
    val_data = pad_sequences(val_data, maxlen = MAX_WORD_LEN)


'''
Build and compile model
'''
model_input = Input(shape = (INPUT_SHAPE,), name = 'input')
char_embed = Embedding(
    input_dim = num_chars, output_dim = STATE_DIM,
    mask_zero = USE_PADDING, name = 'embed'
)(model_input)

if USE_BIDIRECTIONAL:
    # Bi_LSTM stack
    bi_lstm_stack = [
        Bidirectional(LSTM(
            units = STATE_DIM, return_sequences = True,
            dropout = 0.5, recurrent_dropout = 0.5,
            unroll = USE_PADDING,
        ), name = 'bi_lstm_0')(char_embed)
    ]
    for i in range(STACK_SIZE - 1):
        bi_lstm_stack += [
            Bidirectional(LSTM(
                units = STATE_DIM, return_sequences = True,
                dropout = 0.5, recurrent_dropout = 0.5,
                unroll = USE_PADDING,
            ), name = 'bi_lstm_%d' % (i+1))(bi_lstm_stack[i])
        ]
    stack_output = bi_lstm_stack[-1]
else:
    # LSTM stack
    lstm_cells = [
        LSTMCell(
            units = STATE_DIM, #dropout = 0.5, recurrent_dropout = 0.5,
            name = 'lstm_cell_%d' % i
        ) for i in range(STACK_SIZE)
    ]
    lstm_stack = RNN(
        lstm_cells, return_sequences = True, unroll = USE_PADDING, name = 'lstm_stack'
    )(char_embed)
    stack_output = lstm_stack


    
# Output layer
noise_dim = STATE_DIM * (2 if USE_BIDIRECTIONAL else 1)
model_output = TimeDistributed(Dense(
    num_chars, activation = 'softmax', name = 'output'
), name = 'td_output')(Dropout(0.5, noise_shape = (None, 1, noise_dim))(stack_output))

# Build, compile, and summarize model
model = Model(inputs = model_input, outputs = model_output)
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy')
model.summary()

'''
Fit model
'''
# Callback model that performs some name generation tests every few epochs
class GenerativeTest(Callback):
    def __init__(self, num_epochs, num_tests):
        self.num_epochs = num_epochs
        self.num_tests = num_tests
            
    def on_epoch_end(self, epoch, logs = None):
        if epoch == 0 or (epoch + 1) % self.num_epochs == 0:
            gen_in = np.ones((self.num_tests, MAX_WORD_LEN)) * SOF_INDEX
            gen_out = np.ones((self.num_tests, MAX_WORD_LEN)) * EOF_INDEX
            for i in range(MAX_WORD_LEN):
                inputs = gen_in[:, :(i+1)]
                outputs = self.model.predict(inputs)
                # Sample a character from the output distribution of each
                samp_chars = np.ones((self.num_tests,)) * SOF_INDEX
                while any(samp_chars == SOF_INDEX):
                    redo = np.nonzero(samp_chars == SOF_INDEX)
                    samp_chars[redo] = np.apply_along_axis(
                        lambda x: np.random.choice(num_chars, p = x),
                        -1, outputs[redo, -1, :]
                    ).flatten()
                #print('\n   ', outputs.shape, '|', outputs[-1,-1,:].shape, '=>', samp_chars)
                gen_out[:, i] = samp_chars
                if i < MAX_WORD_LEN - 1:
                    gen_in[:, i+1] = samp_chars
                if all(np.apply_along_axis(lambda r: EOF_INDEX in r, 1, gen_out[:,:(i+1)])):
                    break
            # Convert the generated names from index sequences to characters
            names = np.apply_along_axis(
                lambda r: ''.join([i2c[i] for i in r]),
                1, gen_out
            )
            end_indices = [
                MAX_WORD_LEN if EOF_CHAR not in n else n.index(EOF_CHAR) for n in names
            ]
            names = [n[:end_indices[i]] for i, n in enumerate(names)]
            print('\n  Test names: %s' % ('\n              '.join(names)))


# Create some callbacks
reduce_lr = ReduceLROnPlateau(factor = 0.5, patience = 5)
model_checkpoint = ModelCheckpoint(
    SAVE_MODEL_FILE + '.h5', save_best_only = True  #'_e{epoch:d}' + 
)
generative_test = GenerativeTest(5, 3)

# Data generator
def data_gen(data, shuffle = True):
    while True:
        if shuffle:
            np.random.shuffle(data)
        for inputs in data:
            targets = to_categorical(inputs[1:], num_classes = num_chars)
            yield (np.array(inputs[:-1]).T, np.array(targets, ndmin = 3))

model.fit_generator(
    data_gen(data), steps_per_epoch = int(np.ceil(len(data) // BATCH_SIZE)),
    epochs = EPOCHS, validation_data = data_gen(val_data, shuffle = False),
    validation_steps = int(np.ceil(len(val_data) // BATCH_SIZE)),
    callbacks = [
        reduce_lr, generative_test, model_checkpoint,
    ], verbose = 1,
)


'''
Save model and related
'''
model.save(MODEL_NAME + '.h5')
pickle.dump(
    (c2i, i2c, ((SOF_CHAR, SOF_INDEX), (EOF_CHAR, EOF_INDEX))),
    open(DICT_PKL, 'wb')
)
