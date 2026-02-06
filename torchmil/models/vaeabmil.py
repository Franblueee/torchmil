import torch

from .mil_model import MILModel
from torchmil.nn import AttentionPool, LazyLinear
from torchmil.nn.utils import get_feat_dim
from torchmil.nn import VariationalAutoEncoderMIL


class VAEABMIL(MILModel):
    r"""
    Variational Autoencoder - Attention-based Multiple Instance Learning (VAEABMIL) model, proposed in the paper [Using Variational Autoencoders for Out of Distribution Detection in Histological Multiple Instance Learning](https://ieeexplore.ieee.org/abstract/document/11098836/).

    The model jointly trains a Variational Autoencoder (VAE) on instance features to learn a latent representation that is used for attention-based multiple instance learning and to detect out-of-distribution instances and bags.

    Given an input bag $\mathbf{X} = \left[ \mathbf{x}_1, \ldots, \mathbf{x}_N \right]^\top \in \mathbb{R}^{N \times P}$, the model uses  the VAE, to obtain an approximated posterior distribution $p(\mathbf{z} | \mathbf{x})$ for each instance $\mathbf{x}$ in the bag. Then, $\mathbf{X} = [\mathbf{z}_1, \ldots, \mathbf{z}_N] \in \mathbb{R}^{N \times D}$ with $\mathbf{z}_i \sim p(\mathbf{z}_i \mid \mathbf{x}_i)$.

    Lastly, it aggregates the instance features into a bag representation $\mathbf{z} \in \mathbb{R}^{D}$ using the attention-based pooling,

    $$
    \mathbf{z}, \mathbf{f} = \operatorname{AttentionPool}(\mathbf{X}).
    $$

    where $\mathbf{f} \in \mathbb{R}^{N}$ are the attention values.
    See [AttentionPool](../nn/attention/attention_pool.md) for more details on the attention-based pooling.
    The bag representation $\mathbf{z}$ is then fed into a classifier (one linear layer) to predict the bag label.
    """

    def __init__(
        self,
        feat_ext: VariationalAutoEncoderMIL,
        in_shape: tuple = None,
        att_dim: int = 128,
        att_act: str = "tanh",
        gated: bool = False,
        criterion: torch.nn.Module = torch.nn.BCEWithLogitsLoss(),
        vae_loss_reduction: str = "mean",
    ) -> None:
        """
        Arguments:
            feat_ext: Variational Autoencoder used as feature extractor.
            in_shape: Shape of input data expected by the feature extractor (excluding batch dimension). If not provided, it will be lazily initialized.
            att_dim: Attention dimension.
            att_act: Activation function for attention. Possible values: 'tanh', 'relu', 'gelu'.
            gated: If True, use gated attention in the attention pooling.
            criterion: Loss function. By default, Binary Cross-Entropy loss from logits.
            vae_loss_reduction: Reduction method for VAE loss. Possible values: 'sum', 'mean', 'none'.
        """
        super().__init__()
        self.criterion = criterion
        self.vae_loss_reduction = vae_loss_reduction

        self.feat_ext = feat_ext
        if in_shape is not None:
            feat_dim = get_feat_dim(feat_ext, in_shape)
        else:
            feat_dim = None

        self.pool = AttentionPool(
            in_dim=feat_dim, att_dim=att_dim, act=att_act, gated=gated
        )

        self.classifier = LazyLinear(in_features=feat_dim, out_features=1)

    def forward(
        self,
        X: torch.Tensor,
        mask: torch.Tensor = None,
        return_att: bool = False,
        return_latent_repr: bool = False,
        n_samples: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Arguments:
            X: Bag features of shape `(batch_size, bag_size, ...)`.
            mask: Mask of shape `(batch_size, bag_size)`.
            return_att: If True, returns attention values (before normalization) in addition to `Y_pred`.
            return_latent_repr: If True, returns latent representation in addition to `Y_pred`. (Currently not implemented)
            n_samples: Number of Monte Carlo samples to use for the VAE.

        Returns:
            Y_pred: Bag label logits of shape `(batch_size,)`.
            att: Only returned when `return_att=True`. Attention values (before normalization) of shape `(batch_size, bag_size)`.
        """

        X = self.feat_ext(X, n_samples)  # (batch_size, bag_size, n_samples, feat_dim)
        X = X.mean(dim=2)  # (batch_size, bag_size, feat_dim)

        out_pool = self.pool(X, mask, return_att)  # (batch_size, feat_dim)

        if return_att:
            Z, f = out_pool  # (batch_size, feat_dim), (batch_size, bag_size)
        else:
            Z = out_pool  # (batch_size, feat_dim)

        Y_pred = self.classifier(Z)  # (batch_size, 1)
        Y_pred = Y_pred.squeeze(-1)  # (batch_size,)

        if return_att:
            return Y_pred, f
        else:
            return Y_pred

    def compute_loss(
        self,
        Y: torch.Tensor,
        X: torch.Tensor,
        mask: torch.Tensor = None,
        n_samples: int = 1,
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute loss given true bag labels.

        Arguments:
            Y: Bag labels of shape `(batch_size,)`.
            X: Bag features of shape `(batch_size, bag_size, ...)`.
            mask: Mask of shape `(batch_size, bag_size)`.
            n_samples: Number of Monte Carlo samples to use for the VAE loss computation.

        Returns:
            Y_pred: Bag label logits of shape `(batch_size,)`.
            loss_dict: Dictionary containing the loss values. Includes the main criterion loss and VAE losses (VaeELL and VaeKL).
        """

        Y_pred = self.forward(X, mask, return_att=False)
        vae_loss = self.feat_ext.compute_loss(
            X, n_samples=n_samples, reduction=self.vae_loss_reduction
        )

        crit_loss = self.criterion(Y_pred.float(), Y.float())
        crit_name = self.criterion.__class__.__name__

        return Y_pred, {crit_name: crit_loss, **vae_loss}

    def predict(
        self, X: torch.Tensor, mask: torch.Tensor = None, return_inst_pred: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict bag and (optionally) instance labels.

        Arguments:
            X: Bag features of shape `(batch_size, bag_size, ...)`.
            mask: Mask of shape `(batch_size, bag_size)`.
            return_inst_pred: If `True`, returns instance labels predictions (attention values), in addition to bag label predictions.

        Returns:
            Y_pred: Bag label logits of shape `(batch_size,)`.
            y_inst_pred: If `return_inst_pred=True`, returns instance labels predictions (attention values) of shape `(batch_size, bag_size)`.
        """
        return self.forward(X, mask, return_att=return_inst_pred)

    def log_marginal_likelihood_importance_sampling(
        self, X: torch.Tensor, mask: torch.Tensor = None, n_samples: int = 1
    ) -> torch.Tensor:
        """
        Estimate the marginal log-likelihood of the input bag using importance sampling.

        Arguments:
            X: Bag features of shape `(batch_size, bag_size, ...)`.
            n_samples: Number of importance samples to use.

        Returns:
            log_likelihood: Estimated marginal log-likelihood of shape `(batch_size,)`.
        """
        return self.feat_ext.log_marginal_likelihood_importance_sampling(
            X, mask=mask, n_samples=n_samples
        )
