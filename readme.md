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

* [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) (Bahdanau, Cho, Bengio. 2015): recurrent variable-length sequential attention mechanism
* [Attention-Based Models for Speech Recognition](https://arxiv.org/abs/1506.07503) (Chorowski, et al. 2015): provides improvements to the recurrent variable-length sequential attention mechanism
* [Attention is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani, et al. 2017): non-recurrent transformer attention mechanism
* [Self-Attention with Relative Position Representations](https://arxiv.org/abs/1803.02155) (Shaw, Uszkoreit, Vaswani. 2018): provides improvements to the transformer attention mechanism

