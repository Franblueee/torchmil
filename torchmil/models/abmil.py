import torch

from .mil_model import MILModel
from torchmil.nn import AttentionPool, LazyLinear
from torchmil.nn.utils import get_feat_dim


class ABMIL(MILModel):
    r"""
    Attention-based Multiple Instance Learning (ABMIL) model, proposed in the paper [Attention-based Multiple Instance Learning](https://arxiv.org/abs/1802.04712).

    Given an input bag $\mathbf{X} = \left[ \mathbf{x}_1, \ldots, \mathbf{x}_N \right]^\top \in \mathbb{R}^{N \times P}$, the model optionally applies a feature extractor, $\text{FeatExt}(\cdot)$, to transform the instance features: $\mathbf{X} = \text{FeatExt}(\mathbf{X}) \in \mathbb{R}^{N \times D}$.

    Then, it aggregates the instance features into a bag representation $\mathbf{z} \in \mathbb{R}^{D}$ using the attention-based pooling,

    $$
    \mathbf{z}, \mathbf{f} = \operatorname{AttentionPool}(\mathbf{X}).
    $$

    where $\mathbf{f} \in \mathbb{R}^{N}$ are the attention values.
    See [AttentionPool](../nn/attention/attention_pool.md) for more details on the attention-based pooling.
    The bag representation $\mathbf{z}$ is then fed into a classifier (one linear layer) to predict the bag label.
    """

    def __init__(
        self,
        in_shape: tuple = None,
        att_dim: int = 128,
        att_act: str = "tanh",
        gated: bool = False,
        feat_ext: torch.nn.Module = torch.nn.Identity(),
        n_outputs: int = 1,
        criterion: torch.nn.Module = torch.nn.BCEWithLogitsLoss(),
    ) -> None:
        """
        Arguments:
            in_shape: Shape of input data expected by the feature extractor (excluding batch dimension). If not provided, it will be lazily initialized.
            att_dim: Attention dimension.
            att_act: Activation function for attention. Possible values: 'tanh', 'relu', 'gelu'.
            gated: If True, use gated attention in the attention pooling.
            feat_ext: Feature extractor.
            n_outputs: Number of outputs. By default, 1 (binary classification).
            criterion: Loss function. By default, Binary Cross-Entropy loss from logits.
        """
        super().__init__()
        self.num_outputs = n_outputs
        self.criterion = criterion

        self.feat_ext = feat_ext
        if in_shape is not None:
            feat_dim = get_feat_dim(feat_ext, in_shape)
        else:
            feat_dim = None
        self.pool = AttentionPool(
            in_dim=feat_dim, att_dim=att_dim, act=att_act, gated=gated
        )

        self.classifier = LazyLinear(in_features=feat_dim, out_features=n_outputs)

    def forward(
        self, X: torch.Tensor, mask: torch.Tensor = None, return_att: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Arguments:
            X: Bag features of shape `(batch_size, bag_size, ...)`.
            mask: Mask of shape `(batch_size, bag_size)`.
            return_att: If True, returns attention values (before normalization) in addition to `Y_pred`.

        Returns:
            Y_pred: Bag label logits of shape `(batch_size,)`.
            att: Only returned when `return_att=True`. Attention values (before normalization) of shape (batch_size, bag_size).
        """

        X = self.feat_ext(X)  # (batch_size, bag_size, feat_dim)

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
        self, Y: torch.Tensor, X: torch.Tensor, mask: torch.Tensor = None
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute loss given true bag labels.

        Arguments:
            Y: Bag labels of shape `(batch_size,)`.
            X: Bag features of shape `(batch_size, bag_size, ...)`.
            mask: Mask of shape `(batch_size, bag_size)`.

        Returns:
            Y_pred: Bag label logits of shape `(batch_size,)`.
            loss_dict: Dictionary containing the loss value.
        """

        Y_pred = self.forward(X, mask, return_att=False)

        crit_loss = self.criterion(Y_pred.float(), Y.float())
        # crit_loss = self.criterion(Y_pred, Y)
        crit_name = self.criterion.__class__.__name__

        return Y_pred, {crit_name: crit_loss}

    def predict(
        self, X: torch.Tensor, mask: torch.Tensor = None, return_inst_pred: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict bag and (optionally) instance labels.

        Arguments:
            X: Bag features of shape `(batch_size, bag_size, ...)`.
            mask: Mask of shape `(batch_size, bag_size)`.
            return_inst_pred: If `True`, returns instance labels predictions, in addition to bag label predictions.

        Returns:
            Y_pred: Bag label logits of shape `(batch_size,)`.
            y_inst_pred: If `return_inst_pred=True`, returns instance labels predictions of shape `(batch_size, bag_size)`.
        """
        return self.forward(X, mask, return_att=return_inst_pred)
