import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.autograd import Variable

# Starting from sequential data, `batchify` arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.
def batchify(data, batch_size):
    # Work out how cleanly we can divide the dataset into batches
    nbatches = data.size(0) // batch_size
    # Trim off any extra elements that wouldn't cleanly fit
    data = data.narrow(0, 0, nbatches * batch_size)
    # Evenly divide the data across the batches
    data = data.view(batch_size, -1).t().contiguous()
    return data

# Wraps hidden states into new Variables to detach them from their history
def repackage_hidden(h):
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)
    
# `get_batch` subdivides the source data into chunks of the specified length.
# E.g., using the example for the `batchify` function above and a length of 2,
# we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the `batchify` function. The chunks are along dimension 0, corresponding
# to the `seq_len` dimension in the LSTM.
def get_batch(source, i, seq_len, evaluate = False):
    seq_len = min(seq_len, len(source) - 1 - i)
    data = Variable(source[i : i+seq_len], volatile = evaluate)
    target = Variable(source[i+1 : i+1+seq_len].view(-1), volatile = evaluate)
    return data, target

def get_lr_scheduler(h_size, warmup, decay_factor, optimizer):
    '''
    The learning rate scheduler sets the learning rate factor according to:  
    
        lr = d^(-0.5) * min(epoch^(-factor), epoch*warmup^(-(factor+1)))
    
    This corresponds to increasing the learning rate linearly for the first
    `warmup` epochs, then decreasing it proportionally to the inverse
    square root of the epoch number.
    '''
    lrate = lambda e: h_size**(-0.5) * min(
            (e+1)**(-decay_factor), (e+1) * warmup**(-(decay_factor+1))
        )
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lrate)

def visualize_model_parameters(model):
    params = [p for p in model.parameters() if p.grad is not None]
    fig, axs = plt.subplots(1, 2, figsize = (15, 5))

    weights = np.array(torch.cat([w.data.view(-1) for w in params if w.dim() > 1]))
    axs[0].hist(weights, bins = 'auto', density = True, stacked = True)
    axs[0].set_title('Weights (mean: %.3e, var: %.3e)' % (np.mean(weights), np.var(weights)))

    biases = np.array(torch.cat([b.data for b in params if b.dim() == 1]))
    axs[1].hist(biases, bins = 'auto', density = True, stacked = True)
    axs[1].set_title('Biases (mean: %.3e, var: %.3e)' % (np.mean(biases), np.var(biases)))

    plt.show()

'''
Training and evaluation loops
'''
def train(model, train_data, batch_size, seq_len,
          ntokens, criterion, optimizer, lr_scheduler,
          clip = 2, log_interval = 150):
    # Use random length sequences
    seq_lens = []
    tot_len = 0
    jitter = 0.15 * seq_len
    num_data = train_data.size(0)
    while tot_len < num_data - 2:
        if num_data - tot_len - 2 <= seq_len + jitter:
            slen = num_data - tot_len - 2
        else:
            slen = int(np.random.normal(seq_len, jitter))
            if slen <= 0:
                slen = seq_len    # eh
            if tot_len + slen >= num_data - jitter - 2:
                slen = num_data - tot_len - 2
        seq_lens.append(slen)
        tot_len += slen
    i_cumseq = [0] + list(np.cumsum(seq_lens)[:-1])
    idx = np.arange(len(seq_lens))
    np.random.shuffle(idx)
    # Turn on training mode
    model.train(save_wts = False)
    # Initialize RNN states
    states = model.init_states(batch_size)
    # Prep metainfo
    total_loss = 0
    total_epoch_loss = torch.Tensor([0])
    start_time = time.time()
    for batch, i in enumerate(idx):
        # Get training data
        data, targets = get_batch(train_data, i_cumseq[i], seq_lens[i])
        # Repackage the hidden states
        states = repackage_hidden(states)
        # Zero out gradients
        model.zero_grad()
        
        # Run the model forward
        output, _states = model(data, states)
        if np.isnan(output.data).any():
            return 0, total_epoch_loss[0], data, targets, states, _states, 0., 0.
        # Calculate loss
        loss = criterion(output.view(-1, ntokens), targets)
        if np.isnan(loss.data[0]):
            return 1, total_epoch_loss[0], data, targets, states, _states, 0., 0.
        states = _states
        # Propagate loss gradient backwards
        loss.backward()
        # Clip gradients
        inf_norm = max(p.grad.data.abs().max() for p in model.parameters() if p.grad is not None)
        # Experimental -- let the grad norm clip threshold scale with the
        # infinity norm. This will effectively increase the threshold as
        # training progresses.
        total_norm = nn.utils.clip_grad_norm(model.parameters(), max(clip, inf_norm))
        # Scale the batch learning rate so that shorter sequences aren't "stronger"
        scaled_lr = lr_scheduler.get_lr()[0] * np.sqrt(seq_lens[i] / seq_len)
        for param_group in optimizer.param_groups:
            param_group['lr'] = scaled_lr
        # Update parameters
        optimizer.step()
        
        # Get some metainfo
        total_loss += loss.data
        total_epoch_loss += loss.data * data.size(0)
        if batch % log_interval == 0 and batch > 0:
            elapsed = time.time() - start_time
            cur_loss = total_loss[0] / log_interval
            print(' b {:3d}/{:3d} >> {:6.1f} ms/b | lr: {:9.4g} | grad norm: {:5.2f} | inf norm: {:6.3f} | loss: {:4.2f} | perp: {:6.2f}'.format(
                batch, len(seq_lens), elapsed * 1000/log_interval, scaled_lr, total_norm, inf_norm, cur_loss, np.exp(cur_loss)
            ))
            total_loss = 0
            start_time = time.time()
    return -1, total_epoch_loss[0] / num_data, None, None, None, None, total_norm, inf_norm

def evaluate(model, eval_data, eval_batch_size,
             seq_len, ntokens, criterion, save_wts = True):
    model.eval(save_wts = save_wts)
    total_loss = 0
    states = model.init_states(eval_batch_size)
    for i in range(0, eval_data.size(0) - 1, seq_len):
        # Get data
        data, targets = get_batch(eval_data, i, seq_len, evaluate = True)
        # Repackage the hidden states
        states = repackage_hidden(states)
        # Evaluate
        output, states = model(data, states)
        # Calculate loss
        loss = criterion(output.view(-1, ntokens), targets)
        total_loss += loss.data * data.size(0)
    return total_loss[0] / eval_data.size(0)

def train_eval_loop(model, train_data, val_data, batch_size, eval_batch_size,
                    seq_len, ntokens, criterion, eval_criterion, optimizer,
                    lr_scheduler, epochs, warmup_steps, early_stopping = 0,
                    clip = 2, log_interval = 150, ckpt = None):
    '''
    `ckpt`: if a string, indicates the filepath to where saved model checkpoints
    should go.  
    '''
    WIDTH = 108
    CAUSES = ['output', 'grad']
    # Early stopping
    stagnant = 0
    best_train_loss = None
    best_train_loss_epoch = 0
    best_val_loss = None
    best_val_loss_epoch = 0
    better = better_train = better_val = False  # This is just because I'm real extra
    # Keep statistics
    train_loss_hist    = []
    val_loss_hist      = []
    grad_norm_hist     = []
    grad_inf_norm_hist = []
    weight_max_hist    = []
    weight_mean_hist   = []
    weight_var_hist    = []
    bias_max_hist      = []
    bias_mean_hist     = []
    bias_var_hist      = []

    for epoch in range(epochs):
        lr_scheduler.step()
        print('Epoch {:3d}/{:3d}) lr = {:.4g}{}'.format(epoch+1, epochs, np.mean(lr_scheduler.get_lr()[0]), ' (warmup)' if epoch < warmup_steps else ''))
        start_time = time.time()
        stat, train_loss, data, targets, states, nstates, total_norm, inf_norm = train(
            model, train_data, batch_size, seq_len, ntokens,
            criterion, optimizer, lr_scheduler, clip, log_interval
        )
        if stat in list(range(len(CAUSES))):
            c = CAUSES[stat]
            n = (WIDTH - len(c) - 4) // 2
            print('\n' + (' '*n) + 'NaN ' + c)
            break
        elapsed = time.time() - start_time
        val_loss = evaluate(
            model, val_data, eval_batch_size, 
            seq_len, ntokens, eval_criterion,
            save_wts = False
        )
        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)
        # Calculate statistics
        params = [p for p in model.parameters() if p.grad is not None]
        weights = [p for p in params if p.dim() > 1]
        biases = [p for p in params if p.dim() == 1]

        weight_max = max(w.data.abs().max() for w in weights)
        weight_mean = np.mean(torch.cat([w.data.view(-1) for w in weights]))
        weight_var = np.var(torch.cat([w.data.view(-1) for w in weights]))
        bias_max = max(b.data.abs().max() for b in biases)
        bias_mean = np.mean(torch.cat([b.data.view(-1) for b in biases]))
        bias_var = np.var(torch.cat([b.data.view(-1) for b in biases]))

        # Save statistics
        grad_norm_hist.append(total_norm)
        grad_inf_norm_hist.append(inf_norm)
        weight_max_hist.append(weight_max)
        weight_mean_hist.append(weight_mean)
        weight_var_hist.apend(weight_var)
        bias_max_hist.append(bias_max)
        bias_mean_hist.append(bias_mean)
        bias_var_hist.append(bias_var)
        
        # Check for early stopping. Either the training or validation loss has
        # to improve each epoch. If neither does for early_stopping epochs,
        # training ends early
        if best_train_loss is None or train_loss <= best_train_loss:
            best_train_loss = train_loss
            best_train_loss_epoch = epoch
            better = better_train = True
        if best_val_loss is None or val_loss <= best_val_loss:
            best_val_loss = val_loss
            best_val_loss_epoch = epoch
            better = better_val = True

        visualize_model_parameters(model)
        
        print('-' * WIDTH)
        print(
            'Elapsed time: {:7.2f} sec | Grad norm: {:6.3f} | Grad inf. norm: {:7.4f} | train loss, perp: {:5.3f}, {:6.2f} {} | valid loss, perp: {:5.3f}, {:6.2f} {}'.format(
                elapsed, total_norm, inf_norm,
                train_loss, np.exp(train_loss), '  ' if better_train else ':(',
                val_loss, np.exp(val_loss), '  ' if better_val else ':('
        ))
        # Print statistics
        print(
            'Wt max: {:6.3f} | Wt mean: {:6.3f} | Wt var: {:.3e} | Bias max: {:6.3f} | Bias mean: {:6.3f} | Bias var: {:.3e}'.format(
                weight_max, weight_mean, weight_var, bias_max, bias_mean, bias_var
        ))
        print('=' * WIDTH)
        print('\n')
        
        if better:
            stagnant = 0
            better = better_train = better_val = False
            if ckpt is not None:
                torch.save(model, ckpt)
        else:
            stagnant += 1
            if stagnant >= early_stopping and early_stopping > 0:
                break
        # End training loop
    train_stats = {
        'epochs'        : epoch + 1,
        'train_loss'    : train_loss_hist,
        'val_loss'      : val_loss_hist,
        'grad_norm'     : grad_norm_hist,
        'grad_inf_norm' : grad_inf_norm_hist,
        'weight_max'    : weight_max_hist,
        'weight_mean'   : weight_mean_hist,
        'weight_var'    : weight_var_hist,
        'bias_max'      : bias_max_hist,
        'bias_mean'     : bias_mean_hist,
        'bias_var'      : bias_var_hist,
        'best_losses'   : {
            'train' : {
                'loss'  : best_train_loss,
                'epoch' : best_train_loss_epoch + 1,
            },
            'val'   : {
                'loss'  : best_val_loss,
                'epoch' : best_val_loss_epoch + 1,
            },
        },
    }
    return train_stats, stat, train_loss, data, targets, states, nstates
