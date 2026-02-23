import torch
import torch.nn as nn
import numpy as np

from torchmil.models.mil_model import MILModel
from torchmil.nn.utils import get_feat_dim, masked_softmax


class DSMIL(MILModel):
    r"""
    Dual-stream Multiple Instance Learning (DSMIL) model, proposed in the paper [Dual-stream Multiple Instance Learning Network
    for Whole Slide Image Classification with Self-supervised Contrastive Learning](https://arxiv.org/pdf/2011.08939).

    **Overview.**
    Given an input bag $\mathbf{X} = \left[ \mathbf{x}_1, \ldots, \mathbf{x}_N \right]^\top \in \mathbb{R}^{N \times P}$, the model optionally applies a feature extractor, $\text{FeatExt}(\cdot)$, to transform the instance features: $\mathbf{X} = \text{FeatExt}(\mathbf{X}) \in \mathbb{R}^{N \times D}$.

    Then, two streams are used. The **first stream** uses an instance classifier $c \ \colon \mathbb{R}^D \to \mathbb{R}$ (implemented as a linear layer) and retrieves the instance with the highest logit score,

    $$
    m = \arg \max \{ c(\mathbf{x}_1), \ldots, c(\mathbf{x}_N) \}.
    $$

    Then, the **second stream** computes the bag representation $\mathbf{z} \in \mathbb{R}^D$ as

    $$
    \mathbf{z} = \frac{ \exp \left( \mathbf{q}_i^\top \mathbf{q}_m \right)}{\sum_{k=1}^N \exp \left( \mathbf{q}_k^\top \mathbf{q}_m \right)} \mathbf{v}_i,
    $$

    where $\mathbf{q}_i = \mathbf{W}_q \mathbf{x}_i$ and $\mathbf{v}_i = \mathbf{W}_v \mathbf{x}_i$.
    This is similar to self-attention with the difference that query-key matching is performed only with the critical instance.

    Finally, the bag representation is used to predict the bag label using a bag classifier implemented as a linear layer.

    **Loss function.**
    By default, the model is trained end-to-end using the followind per-bag loss:

    $$
    \ell = \ell_{\text{BCE}}(Y, \hat{Y}) + \ell_{\text{BCE}}(Y, c(\mathbf{x}_m)),
    $$

    where $\ell_{\text{BCE}}$ is the Binary Cross-Entropy loss, $Y$ is the true bag label, $\hat{Y}$ is the predicted bag label, and $c(\mathbf{x}_m)$ is the predicted label of the critical instance.

    """

    def __init__(
        self,
        in_shape: tuple = None,
        att_dim: int = 128,
        nonlinear_q: bool = False,
        nonlinear_v: bool = False,
        dropout: float = 0.0,
        feat_ext: torch.nn.Module = torch.nn.Identity(),
        n_outputs: int = 1,
        criterion: torch.nn.Module = torch.nn.BCEWithLogitsLoss(),
    ) -> None:
        """
        Arguments:
            in_shape: Shape of input data expected by the feature extractor (excluding batch dimension).
            att_dim: Attention dimension.
            nonlinear_q: If True, apply nonlinearity to the query.
            nonlinear_v: If True, apply nonlinearity to the value.
            dropout: Dropout rate.
            feat_ext: Feature extractor.
            n_outputs: Number of outputs. By default, 1 (binary classification).
            criterion: Loss function. By default, Binary Cross-Entropy loss from logits.
        """
        super(DSMIL, self).__init__()
        self.num_outputs = n_outputs
        self.criterion = criterion
        self.feat_ext = feat_ext

        feat_dim = get_feat_dim(feat_ext, in_shape)

        if nonlinear_q:
            self.q_nn = nn.Sequential(
                nn.Linear(feat_dim, att_dim),
                nn.ReLU(),
                nn.Linear(att_dim, att_dim),
                nn.Tanh(),
            )
        else:
            self.q_nn = nn.Linear(feat_dim, att_dim)

        if nonlinear_v:
            self.v_nn = nn.Sequential(
                nn.Dropout(dropout), nn.Linear(feat_dim, feat_dim), nn.ReLU()
            )
        else:
            self.v_nn = nn.Identity()

        self.inst_classifier = nn.Linear(feat_dim, 1)
        self.bag_classifier = nn.Linear(feat_dim, n_outputs)

    def forward(
        self,
        X: torch.Tensor,
        mask: torch.Tensor = None,
        return_att: bool = False,
        return_inst_pred: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Arguments:
            X: Bag features of shape `(batch_size, bag_size, ...)`.
            mask: Mask of shape `(batch_size, bag_size)`.
            return_att: If True, returns attention values (before normalization) in addition to `Y_pred`.
            return_inst_pred: If True, returns instance label logits in addition to `Y_pred`.

        Returns:
            Y_pred: Bag label logits of shape `(batch_size,)`.
            att: Only returned when `return_att=True`. Attention values (before normalization) of shape (batch_size, bag_size).
            y_pred: Only returned when `return_inst_pred=True`. Instance label logits of shape `(batch_size, bag_size)`.
        """

        if mask is not None:
            mask = mask.bool()

        X = self.feat_ext(X)  # (batch_size, bag_size, feat_dim)

        y_logits = self.inst_classifier(X)  # (batch_size, bag_size, 1)

        V = self.v_nn(X)  # (batch_size, bag_size, feat_dim)
        Q = self.q_nn(X)  # (batch_size, bag_size, att_dim)

        # sort class scores along the instance dimension
        indices_max = torch.sort(y_logits, 1, descending=True)[
            1
        ]  # (batch_size, bag_size, 1)
        idx_max = indices_max[:, 0, :]  # (batch_size, 1)
        idx_max = idx_max.unsqueeze(-1).expand(
            -1, -1, Q.size(-1)
        )  # (batch_size, 1, att_dim)

        # compute queries of critical instances
        Q_max = torch.gather(Q, 1, idx_max)  # (batch_size, 1, att_dim)

        # compute inner product of Q to each entry of q_max
        A = torch.bmm(Q, Q_max.transpose(1, 2))  # (batch_size, bag_size, 1)

        # scale and normalize the attention scores
        scale = np.sqrt(Q_max.size(-1))
        A = A / scale

        A = masked_softmax(A, mask)  # (batch_size, bag_size, 1)

        # compute bag representation
        z = torch.bmm(A.transpose(1, 2), V)  # (batch_size, 1, feat_dim)

        Y_pred = self.bag_classifier(z)  # (batch_size, 1, n_outputs)
        if self.num_outputs == 1:
            Y_pred = Y_pred.squeeze()  # (batch_size,)
        else:
            Y_pred = Y_pred.squeeze(1)  # (batch_size, n_outputs)

        y_logits = y_logits.squeeze(-1)  # (batch_size, bag_size)

        if return_att:
            if return_inst_pred:
                return Y_pred, A, y_logits
            else:
                return Y_pred, A
        else:
            if return_inst_pred:
                return Y_pred, y_logits
            else:
                return Y_pred

    def compute_loss(
        self,
        Y: torch.Tensor,
        X: torch.Tensor,
        mask: torch.Tensor = None,
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
        Y_pred, y_pred = self.forward(X, mask, return_inst_pred=True)
        max_pred, _ = torch.max(y_pred, 1)  # (batch_size,)
        bag_pred = 0.5 * (Y_pred + max_pred)  # (batch_size,)
        crit_loss = self.criterion(Y_pred.float(), Y.float())
        crit_name = self.criterion.__class__.__name__
        max_loss = self.criterion(max_pred.float(), Y.float())
        return bag_pred, {crit_name: crit_loss, f"{crit_name}_max": max_loss}

    def predict(
        self, X: torch.Tensor, mask: torch.Tensor = None, return_inst_pred: bool = False
    ) -> torch.Tensor:
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
        Y_pred, y_logits_pred = self.forward(X, mask, return_inst_pred=True)
        max_pred, _ = torch.max(y_logits_pred, 1)  # (batch_size,)
        bag_pred = 0.5 * (Y_pred + max_pred)  # (batch_size,)
        if return_inst_pred:
            return bag_pred, y_logits_pred
        else:
            return bag_pred
