from typing import Union

import torch

from torchmil.models.mil_model import MILModel, compute_criterion_loss, init_criterion, squeeze_binary_logits

from torchmil.nn import SmAttentionPool

from torchmil.nn.utils import get_feat_dim


class SmABMIL(MILModel):
    r"""
    Attention-based Multiple Instance Learning (ABMIL) model with the $\texttt{Sm}$ operator.
    Proposed in [Sm: enhanced localization in Multiple Instance Learning for medical imaging classification](https://arxiv.org/abs/2410.03276).

    Given an input bag $\mathbf{X} = \left[ \mathbf{x}_1, \ldots, \mathbf{x}_N \right]^\top \in \mathbb{R}^{N \times P}$, the model optionally applies a feature extractor, $\text{FeatExt}(\cdot)$, to transform the instance features: $\mathbf{X} = \text{FeatExt}(\mathbf{X}) \in \mathbb{R}^{N \times D}$.

    Then, it aggregates the instance features into a bag representation $\mathbf{z} \in \mathbb{R}^{D}$ using an attention-based pooling mechanism that incorporates the $\texttt{Sm}$ operator,

    $$
    \mathbf{z}, \mathbf{f} = \operatorname{SmAttentionPool}(\mathbf{X}),
    $$

    where $\mathbf{f} \in \mathbb{R}^{N}$ are the attention values.
    See [SmAttentionPool](../nn/attention/sm_attention_pool.md) for more details on the pooling operator
    The bag representation $\mathbf{z}$ is then fed into a classifier (one linear layer) to predict the bag label.
    """

    def __init__(
        self,
        in_shape: tuple,
        att_dim: int = 128,
        att_act: str = "tanh",
        sm_mode: str = "approx",
        sm_alpha: Union[float, str] = "trainable",
        sm_steps: int = 10,
        sm_where: str = "early",
        spectral_norm: bool = False,
        feat_ext: torch.nn.Module = torch.nn.Identity(),
        n_outputs: int = 1,
        criterion: torch.nn.Module = torch.nn.BCEWithLogitsLoss(),
    ) -> None:
        """
        Arguments:
            in_shape: Shape of input data expected by the feature extractor (excluding batch dimension).
            att_dim (int): Attention dimension.
            att_act (str): Activation function for attention. Possible values: 'tanh', 'relu', 'gelu'.
            sm_mode (str): Mode for the Sm operator. Possible values: 'approx', 'exact'.
            sm_alpha (Union[float, str]): Alpha value for the Sm operator. If 'trainable', alpha is trainable.
            sm_steps (int): Number of steps for the Sm operator. Only used if `sm_mode='approx'`.
            sm_where (str): Where to apply the Sm operator. Possible values: 'early', 'mid', 'late'.
            spectral_norm (bool): If True, apply spectral normalization to all linear layers.
            feat_ext (torch.nn.Module): Feature extractor.
            n_outputs: Number of outputs. By default, 1 (binary classification).
            criterion (torch.nn.Module): Loss function. By default, Binary Cross-Entropy loss from logits for binary classification.
        """
        super().__init__()
        self.num_outputs = n_outputs

        self.feat_ext = feat_ext
        feat_dim = get_feat_dim(feat_ext, in_shape)

        self.pool = SmAttentionPool(
            in_dim=feat_dim,
            att_dim=att_dim,
            act=att_act,
            sm_mode=sm_mode,
            sm_alpha=sm_alpha,
            sm_steps=sm_steps,
            sm_where=sm_where,
            spectral_norm=spectral_norm,
        )
        self.classifier = torch.nn.Linear(feat_dim, n_outputs)
        self.last_layer = self.classifier

        self.criterion = init_criterion(n_outputs, criterion)

    def forward(
        self,
        X: torch.Tensor,
        adj: torch.Tensor,
        mask: torch.Tensor = None,
        return_att: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Arguments:
            X: Bag features of shape `(batch_size, bag_size, ...)`.
            adj: Adjacency matrix of shape `(batch_size, bag_size, bag_size)`.
            mask: Mask of shape `(batch_size, bag_size)`.
            return_att: If True, returns attention values (before normalization) in addition to `Y_pred`.

        Returns:
            Y_pred: Bag label logits of shape `(batch_size,)`.
            att: Only returned when `return_att=True`. Attention values (before normalization) of shape (batch_size, bag_size).
        """

        X = self.feat_ext(X)  # (batch_size, bag_size, feat_dim)

        out_pool = self.pool(X, adj, mask, return_att)  # (batch_size, feat_dim)

        if return_att:
            Z, f = out_pool  # (batch_size, feat_dim), (batch_size, bag_size)
        else:
            Z = out_pool  # (batch_size, feat_dim)

        Y_pred = self.classifier(Z)  # (batch_size, 1)
        Y_pred = squeeze_binary_logits(Y_pred, self.num_outputs)

        if return_att:
            return Y_pred, f
        else:
            return Y_pred

    def compute_loss(
        self,
        Y: torch.Tensor,
        X: torch.Tensor,
        adj: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute loss given true bag labels.

        Arguments:
            Y: Bag labels of shape `(batch_size,)`.
            X: Bag features of shape `(batch_size, bag_size, ...)`.
            adj: Adjacency matrix of shape `(batch_size, bag_size, bag_size)`.
            mask: Mask of shape `(batch_size, bag_size)`.

        Returns:
            Y_pred: Bag label logits of shape `(batch_size,)`.
            loss_dict: Dictionary containing the loss value.
        """

        Y_pred = self.forward(X, adj, mask, return_att=False)

        crit_loss = compute_criterion_loss(self.criterion, Y_pred, Y, self.num_outputs)
        crit_name = self.criterion.__class__.__name__

        return Y_pred, {crit_name: crit_loss}

    def predict(
        self,
        X: torch.Tensor,
        adj: torch.Tensor,
        mask: torch.Tensor = None,
        return_inst_pred: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict bag and (optionally) instance labels.

        Arguments:
            X: Bag features of shape `(batch_size, bag_size, ...)`.
            adj: Adjacency matrix of shape `(batch_size, bag_size, bag_size)`.
            mask: Mask of shape `(batch_size, bag_size)`.
            return_inst_pred (bool): If `True`, returns instance labels predictions, in addition to bag label predictions.

        Returns:
            Y_pred: Bag label logits of shape `(batch_size,)`.
            y_inst_pred: If `return_inst_pred=True`, returns instance labels predictions of shape `(batch_size, bag_size)`.
        """
        return self.forward(X, adj, mask, return_att=return_inst_pred)
