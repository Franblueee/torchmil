# torchmil.nn

**torchmil.nn** is a collection of PyTorch modules frequently used in Multiple Instance Learning (MIL) models.
These modules are designed to be flexible and easy to use, allowing you to build custom MIL models with ease.

## Available modules
- [Attention](attention/index.md)
    - [Attention Pool](attention/attention_pool.md)
    - [Sm Attention Pool](attention/sm_attention_pool.md)
    - [Probabilistic Smooth Attention Pool](attention/prob_smooth_attention_pool.md)
    - [Multihead Self-Attention](attention/multihead_self_attention.md)
    - [Multihead Self-Attention with Relative Positional Encoding (iRPE)](attention/irpe_multihead_self_attention.md)
    - [Nystrom Attention](attention/nystrom_attention.md)
    - [Multihead Cross-Attention](attention/multihead_cross_attention.md)
- [Graph Neural Networks (GNNs)](gnns/index.md)
    - [Deep Graph Convolutional Network (DeepGCN) layer](gnns/deepgcn.md)
    - [Graph Convolutional Network (GCN) convolution](gnns/gcn_conv.md)
    - [Dense MinCut pooling](gnns/dense_mincut_pool.md)
    - [Chebyshev spectral graph convolution (ChebConv)](gnns/chebconv.md)
- [Transformers](transformers/index.md)
    - [Transformer base class](transformers/base_transformer.md)
    - [Conventional Transformer](transformers/conventional_transformer.md)
    - [Sm Transformer](transformers/sm_transformer.md)
    - [Nystrom Transformer](transformers/nystrom_transformer.md)
    - [Transformer with image Relative Positional Encoding (iRPE)](transformers/irpe_transformer.md)
    - [Tokens-2-Token](transformers/t2t.md)
- [Sm operator](sm.md)
- [Max Pool](max_pool.md)
- [Mean Pool](mean_pool.md)
- [Multi Layer Perceptron](mlp.md)
- [Variational Autoencoder](variational_autoencoder.md)
