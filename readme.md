# Sequential Attention mechanisms

Repo of sequential attention mechanisms from various papers, including their
implementations and studies of how they behave within various model architectures.

## Using and viewing

The implementations themselves are primarily done in [PyTorch](https://pytorch.org/) (v0.3 for now).
They're built in a way that lets them be easily imported and used into any model
following the documented input/output shapes.

The tests and studies of the implementations are done in Jupyter notebooks and
can be viewed without having PyTorch installed (but not runnable without it). 

## References

* [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) (Bahdanau, Cho, Bengio. 2015): recurrent variable-length sequential attention
* [Attention is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani, et al. 2017): non-recurrent transformer attention

