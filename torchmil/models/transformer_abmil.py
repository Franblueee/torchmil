import torch

from .mil_model import MILModel

from torchmil.nn import AttentionPool, TransformerEncoder
from torchmil.nn.utils import get_feat_dim


class TransformerABMIL(MILModel):
    r"""
    Transformer Attention-based Multiple Instance Learning model.
    Proposed in [Sm: enhanced localization in Multiple Instance Learning for medical imaging classification](https://arxiv.org/abs/2410.03276).

    Given an input bag $\mathbf{X} = \left[ \mathbf{x}_1, \ldots, \mathbf{x}_N \right]^\top \in \mathbb{R}^{N \times P}$, the model optionally applies a feature extractor, $\text{FeatExt}(\cdot)$, to transform the instance features: $\mathbf{X} = \text{FeatExt}(\mathbf{X}) \in \mathbb{R}^{N \times D}$.
    Then, it transforms the instance features using a transformer encoder,

    $$ \mathbf{X} = \text{TransformerEncoder}(\mathbf{X}) \in \mathbb{R}^{N \times D}, $$

    and finally it aggregates the instance features into a bag representation $\mathbf{z} \in \mathbb{R}^{D}$ using the attention-based pooling,

    $$
    \mathbf{z}, \mathbf{f} = \operatorname{AttentionPool}(\mathbf{X}).
    $$

    where $\mathbf{f} = \operatorname{MLP}(\mathbf{X}) \in \mathbb{R}^{N}$ are the attention values.
    The bag representation $\mathbf{z}$ is then fed into a classifier (one linear layer) to predict the bag label.

    See [AttentionPool](../nn/attention/attention_pool.md) for more details on the attention-based pooling, and [TransformerEncoder](../nn/transformers/conventional_transformer.md) for more details on the transformer encoder.
    """

    def __init__(
        self,
        in_shape: tuple,
        pool_att_dim: int = 128,
        pool_act: str = "tanh",
        pool_gated: bool = False,
        feat_ext: torch.nn.Module = torch.nn.Identity(),
        transf_att_dim: int = 512,
        transf_n_layers: int = 1,
        transf_n_heads: int = 8,
        transf_use_mlp: bool = True,
        transf_add_self: bool = True,
        transf_dropout: float = 0.0,
        criterion: torch.nn.Module = torch.nn.BCEWithLogitsLoss(),
    ) -> None:
        """
        Class constructor.

        Arguments:
            in_shape: Shape of input data expected by the feature extractor (excluding batch dimension). If not provided, it will be lazily initialized.
            pool_att_dim: Attention dimension for pooling.
            pool_act: Activation function for pooling. Possible values: 'tanh', 'relu', 'gelu'.
            pool_gated: If True, use gated attention in the attention pooling.
            feat_ext: Feature extractor.
            transf_att_dim: Attention dimension for transformer encoder.
            transf_n_layers: Number of layers in transformer encoder.
            transf_n_heads: Number of heads in transformer encoder.
            transf_use_mlp: Whether to use MLP in transformer encoder.
            transf_add_self: Whether to add input to output in transformer encoder.
            transf_dropout: Dropout rate in transformer encoder.
            criterion: Loss function. By default, Binary Cross-Entropy loss from logits for binary classification.
        """
        super().__init__()
        self.criterion = criterion

        self.feat_ext = feat_ext
        feat_dim = get_feat_dim(feat_ext, in_shape)
        self.transformer_encoder = TransformerEncoder(
            in_dim=feat_dim,
            att_dim=transf_att_dim,
            n_layers=transf_n_layers,
            n_heads=transf_n_heads,
            use_mlp=transf_use_mlp,
            add_self=transf_add_self,
            dropout=transf_dropout,
        )
        self.pool = AttentionPool(
            in_dim=feat_dim, att_dim=pool_att_dim, act=pool_act, gated=pool_gated
        )
        self.last_layer = torch.nn.Linear(feat_dim, 1)

    def forward(
        self, X: torch.Tensor, mask: torch.Tensor = None, return_att: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Arguments:
            X: Bag features of shape `(batch_size, bag_size, ...)`.
            mask: Mask of shape `(batch_size, bag_size)`.
            return_att: If True, returns attention values (before normalization) in addition to `Y_logits_pred`.

        Returns:
            Y_pred: Bag label logits of shape `(batch_size,)`.
            att: Only returned when `return_att=True`. Attention values (before normalization) of shape (batch_size, bag_size).
        """

        X = self.feat_ext(X)  # (batch_size, bag_size, feat_dim)

        X = self.transformer_encoder(X, mask)  # (batch_size, bag_size, feat_dim)

        out_pool = self.pool(X, mask, return_att=return_att)
        if return_att:
            z, f = out_pool  # z: (batch_size, emb_dim), f: (batch_size, bag_size)
        else:
            z = out_pool  # (batch_size, emb_dim)

        Y_pred = self.last_layer(z)  # (batch_size, n_samples, 1)
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
