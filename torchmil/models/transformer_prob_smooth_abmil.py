import torch
from torch import Tensor

from torchmil.models.mil_model import MILModel
from torchmil.nn import ProbSmoothAttentionPool, TransformerEncoder
from torchmil.nn.utils import get_feat_dim, LazyLinear


class TransformerProbSmoothABMIL(MILModel):
    r"""
    Transformer Attention-based Multiple Instance Learning model, with probabilistic attention-based pooling.
    Proposed in [Probabilistic Smooth Attention for Deep Multiple Instance Learning in Medical Imaging](https://arxiv.org/abs/2507.14932).

    Given an input bag $\mathbf{X} = \left[ \mathbf{x}_1, \ldots, \mathbf{x}_N \right]^\top \in \mathbb{R}^{N \times P}$, the model optionally applies a feature extractor, $\text{FeatExt}(\cdot)$, to transform the instance features: $\mathbf{X} = \text{FeatExt}(\mathbf{X}) \in \mathbb{R}^{N \times D}$.
    Then, it transforms the instance features using a transformer encoder,

    $$ \mathbf{X} = \text{TransformerEncoder}(\mathbf{X}) \in \mathbb{R}^{N \times D}. $$

    Subsequently, it aggregates the instance features into a bag representation using a probabilistic attention-based pooling mechanism, as detailed in [ProbSmoothAttentionPool](../nn/attention/prob_smooth_attention_pool.md).

    Specifically, it computes a mean vector $\mathbf{\mu}_{\mathbf{f}} \in \mathbb{R}^N$ and a variance vector $\mathbf{\sigma}_{\mathbf{f}^2} \in \mathbb{R}^N$ that define the attention distribution $q(\mathbf{f} \mid \mathbf{X}) = \mathcal{N}\left(\mathbf{f} \mid \mathbf{\mu}_{\mathbf{f}}, \operatorname{diag}(\mathbf{\sigma}_{\mathbf{f}}^2) \right)$,

    $$
    \mathbf{\mu}_{\mathbf{f}}, \mathbf{\sigma}_{\mathbf{f}} = \operatorname{ProbSmoothAttentionPool}(\mathbf{X}).
    $$

    If `covar_mode='zero'`, the variance vector $\mathbf{\sigma}_{\mathbf{f}}^2$ is set to zero, resulting in a deterministic attention distribution.

    Then, $m$ attention vectors $\widehat{\mathbf{F}} = \left[ \widehat{\mathbf{f}}^{(1)}, \ldots, \widehat{\mathbf{f}}^{(m)} \right]^\top \in \mathbb{R}^{m \times N}$ are sampled from the attention distribution. The bag representation $\widehat{\mathbf{z}} \in \mathbb{R}^{m \times D}$ is then computed as:

    $$
    \widehat{\mathbf{z}} = \operatorname{Softmax}(\widehat{\mathbf{F}}) \mathbf{X}.
    $$

    The bag representation $\widehat{\mathbf{z}}$ is fed into a classifier, implemented as a linear layer, to produce bag label predictions $Y_{\text{pred}} \in \mathbb{R}^{m}$.

    Notably, the attention distribution naturally induces a distribution over the bag label predictions. This model thus generates multiple predictions for each bag, corresponding to different samples from this distribution.


    **Regularization.**
    The probabilistic pooling mechanism introduces a regularization term to the loss function that encourages *smoothness* in the attention values.
    Given an input bag $\mathbf{X} = \left[ \mathbf{x}_1, \ldots, \mathbf{x}_N \right]^\top \in \mathbb{R}^{N \times P}$ with adjacency matrix $\mathbf{A} \in \mathbb{R}^{N \times N}$, the regularization term corresponds to

    $$
        \ell_{\text{KL}}(\mathbf{X}, \mathbf{A}) =
            \begin{cases}
                \mathbf{\mu}_{\mathbf{f}}^\top \mathbf{L} \mathbf{\mu}_{\mathbf{f}} \quad & \text{if } \texttt{covar_mode='zero'}, \\
                \mathbf{\mu}_{\mathbf{f}}^\top \mathbf{L} \mathbf{\mu}_{\mathbf{f}} + \operatorname{Tr}(\mathbf{L} \mathbf{\Sigma}_{\mathbf{f}}) - \frac{1}{2}\log \det( \mathbf{\Sigma}_{\mathbf{f}} ) + \operatorname{const} \quad & \text{if } \texttt{covar_mode='diag'}, \\
            \end{cases}
    $$

    where $\operatorname{const}$ is a constant term that does not depend on the parameters, $\mathbf{\Sigma}_{\mathbf{f}} = \operatorname{diag}(\mathbf{\sigma}_{\mathbf{f}}^2)$, $\mathbf{L} = \mathbf{D} - \mathbf{A}$ is the graph Laplacian matrix, and $\mathbf{D}$ is the degree matrix of $\mathbf{A}$.
    This term is then averaged for all bags in the batch and added to the loss function.
    """

    def __init__(
        self,
        in_shape: tuple = None,
        pool_att_dim: int = 128,
        covar_mode: str = "diag",
        n_samples_train: int = 1000,
        n_samples_test: int = 5000,
        feat_ext: torch.nn.Module = torch.nn.Identity(),
        transf_att_dim: int = 512,
        transf_n_layers: int = 1,
        transf_n_heads: int = 8,
        transf_use_mlp: bool = True,
        transf_add_self: bool = True,
        transf_dropout: float = 0.0,
        n_outputs: int = 1,
        criterion: torch.nn.Module = torch.nn.BCEWithLogitsLoss(),
    ) -> None:
        """
        Arguments:
            in_shape: Shape of input data expected by the feature extractor (excluding batch dimension). If not provided, it will be lazily initialized.
            pool_att_dim: Attention dimension.
            covar_mode: Covariance mode for the Gaussian prior. Possible values: 'diag', 'full'.
            n_samples_train: Number of samples for training.
            n_samples_test: Number of samples for testing.
            feat_ext: Feature extractor.
            transf_att_dim: Attention dimension for transformer encoder.
            transf_n_layers: Number of layers in transformer encoder.
            transf_n_heads: Number of heads in transformer encoder.
            transf_use_mlp: Whether to use MLP in transformer encoder.
            transf_add_self: Whether to add input to output in transformer encoder.
            transf_dropout: Dropout rate in transformer encoder.
            n_outputs: Number of outputs. By default, 1 (binary classification).
            criterion: Loss function. By default, Binary Cross-Entropy loss from logits for binary classification.

        """
        super().__init__()
        self.num_outputs = n_outputs
        self.criterion = criterion

        self.feat_ext = feat_ext
        if in_shape is not None:
            feat_dim = get_feat_dim(feat_ext, in_shape)
        else:
            feat_dim = None

        self.transformer_encoder = TransformerEncoder(
            in_dim=feat_dim,
            att_dim=transf_att_dim,
            n_layers=transf_n_layers,
            n_heads=transf_n_heads,
            use_mlp=transf_use_mlp,
            add_self=transf_add_self,
            dropout=transf_dropout,
        )

        self.pool = ProbSmoothAttentionPool(
            in_dim=feat_dim,
            att_dim=pool_att_dim,
            covar_mode=covar_mode,
            n_samples_train=n_samples_train,
            n_samples_test=n_samples_test,
        )
        self.classifier = LazyLinear(feat_dim, n_outputs)

    def forward(
        self,
        X: Tensor,
        adj: Tensor,
        mask: Tensor = None,
        return_att: bool = False,
        return_samples: bool = False,
        return_kl_div: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass.

        Arguments:
            X: Bag features of shape `(batch_size, bag_size, ...)`.
            mask: Mask of shape `(batch_size, bag_size)`.
            adj: Adjacency matrix of shape `(batch_size, bag_size, bag_size)`. Only required when `return_kl_div=True`.
            return_att: If True, returns attention values (before normalization) in addition to `Y_pred`.
            return_samples: If True and `return_att=True`, the attention values returned are samples from the attention distribution.
            return_kl_div: If True, returns the KL divergence between the attention distribution and the prior distribution.

        Returns:
            Y_pred: Bag label logits of shape `(batch_size, n_samples)` if `return_samples=True`, else `(batch_size,)`.
            att: Only returned when `return_att=True`. Attention values (before normalization) of shape `(batch_size, bag_size, n_samples)` if `return_samples=True`, else `(batch_size, bag_size)`.
            kl_div: Only returned when `return_kl_div=True`. KL divergence between the attention distribution and the prior distribution of shape `()`.
        """

        X = self.feat_ext(X)  # (batch_size, bag_size, feat_dim)

        X = self.transformer_encoder(X, mask)  # (batch_size, bag_size, feat_dim)

        out_pool = self.pool(
            X, adj, mask, return_att_samples=return_att, return_kl_div=return_kl_div
        )

        if return_kl_div:
            if return_att:
                z, f, kl_div = out_pool
            else:
                z, kl_div = out_pool
        else:
            if return_att:
                z, f = out_pool
            else:
                z = out_pool

        z = z.transpose(1, 2)  # (batch_size, n_samples, feat_dim)
        Y_pred = self.classifier(z)  # (batch_size, n_samples, n_outputs)
        if self.num_outputs == 1:
            Y_pred = Y_pred.squeeze(-1)  # (batch_size, n_samples)

        if not return_samples:
            Y_pred = Y_pred.mean(dim=1)  # (batch_size,) or (batch_size, n_outputs)
            if return_att:
                f = f.mean(dim=-1)  # (batch_size, bag_size)

        if return_kl_div:
            if return_att:
                return Y_pred, f, kl_div
            else:
                return Y_pred, kl_div
        else:
            if return_att:
                return Y_pred, f
            else:
                return Y_pred

    def compute_loss(
        self, Y: Tensor, X: Tensor, adj: Tensor, mask: Tensor = None
    ) -> tuple[Tensor, dict]:
        """
        Compute loss given true bag labels.

        Arguments:
            Y: Bag labels of shape `(batch_size,)`.
            X: Bag features of shape `(batch_size, bag_size, ...)`.
            mask: Mask of shape `(batch_size, bag_size)`.

        Returns:
            Y_pred: Bag label logits of shape `(batch_size,)`.
            loss_dict: Dictionary containing the loss value and the KL divergence between the attention distribution and the prior distribution.
        """

        Y_pred, kl_div = self.forward(
            X, adj, mask, return_att=False, return_samples=True, return_kl_div=True
        )  # (batch_size, n_samples)
        Y_pred_mean = Y_pred.mean(dim=-1)  # (batch_size,)

        Y = Y.unsqueeze(-1).expand(-1, Y_pred.shape[-1])
        crit_loss = self.criterion(Y_pred.float(), Y.float())
        crit_name = self.criterion.__class__.__name__

        return Y_pred_mean, {crit_name: crit_loss, "KLDiv": kl_div}

    def predict(
        self,
        X: Tensor,
        adj: Tensor,
        mask: Tensor = None,
        return_inst_pred: bool = True,
        return_samples: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """
        Predict bag and (optionally) instance labels.

        Arguments:
            X: Bag features of shape `(batch_size, bag_size, ...)`.
            mask: Mask of shape `(batch_size, bag_size)`.
            return_inst_pred: If True, returns the attention values as instance labels predictions, in addition to bag label predictions.
            return_samples: If True and `return_inst_pred=True`, the instance label predictions returned are samples from the instance label distribution.

        Returns:
            Y_pred: Bag label logits of shape `(batch_size,)`.
            y_inst_pred: Only returned when `return_inst_pred=True`. Attention values (before normalization) of shape `(batch_size, bag_size)` if `return_samples=False`, else `(batch_size, bag_size, n_samples)`.
        """
        return self.forward(
            X, adj, mask, return_att=return_inst_pred, return_samples=return_samples
        )
