import torch

from torchmil.nn import AttentionPool, GCNConv, DeepGCNLayer
from torchmil.models.mil_model import compute_criterion_loss, init_criterion, squeeze_binary_logits

from torchmil.nn.utils import get_feat_dim, LazyLinear


class PatchGCN(torch.nn.Module):
    r"""
    PatchGCN model, as proposed in [Whole Slide Images are 2D Point Clouds: Context-Aware Survival Prediction using Patch-based Graph Convolutional Networks](https://arxiv.org/abs/2107.13048).

    Given an input bag $\mathbf{X} = \left[ \mathbf{x}_1, \ldots, \mathbf{x}_N \right]^\top \in \mathbb{R}^{N \times P}$ with adjacency matrix $\mathbf{A} \in \mathbb{R}^{N \times N}$,
    the model optionally applies a feature extractor, $\text{FeatExt}(\cdot)$, to transform the instance features: $\mathbf{X} = \text{FeatExt}(\mathbf{X}) \in \mathbb{R}^{N \times D}$.

    Then, a Graph Convolutional Network (GCN) and a Multi-Layer Perceptron (MLP) are used to transform the instance features,

    \begin{gather}
    \mathbf{H} = \operatorname{GCN}(\mathbf{X}, \mathbf{A}) \in \mathbb{R}^{N \times \texttt{out_gcn_dim}}, \\
    \mathbf{H} = \operatorname{MLP}(\mathbf{H}) \in \mathbb{R}^{N \times \texttt{hidden_dim}},
    \end{gather}

    where $\texttt{out_gcn_dim} = \texttt{hidden_dim} \cdot \texttt{n_gcn_layers}$.
    These GCNs are implemented using the DeepGCN layer (see [DeepGCNLayer](../nn/gnns/deepgcn.md)) with GCNConv, LayerNorm, and ReLU activation (see [GCNConv](../nn/gnns/gcn_conv.md)),
    along with residual connections and dense connections.

    Then, attention values $\mathbf{f} \in \mathbb{R}^{N \times 1}$ and the bag representation $\mathbf{z} \in \mathbb{R}^{\texttt{hidden_dim}}$
    are computed using the attention pooling mechanism (see [Attention Pooling](../nn/attention/attention_pool.md)),

    \begin{equation}
    \mathbf{z}, \mathbf{f} = \operatorname{AttentionPool}(\mathbf{H}).
    \end{equation}

    Finally, the bag representation $\mathbf{z}$ is fed into a classifier (one linear layer) to predict the bag label.
    """

    def __init__(
        self,
        in_shape: tuple,
        n_gcn_layers: int = 4,
        mlp_depth: int = 1,
        hidden_dim: int = None,
        att_dim: int = 128,
        dropout: float = 0.0,
        feat_ext: torch.nn.Module = torch.nn.Identity(),
        n_outputs: int = 1,
        criterion: torch.nn.Module = torch.nn.BCEWithLogitsLoss(),
    ):
        """
        Arguments:
            in_shape: Shape of input data expected by the feature extractor (excluding batch dimension).
            n_gcn_layers: Number of GCN layers.
            mlp_depth: Number of layers in the MLP (applied after the GCN).
            hidden_dim: Hidden dimension. If not provided, it will be set to the feature dimension.
            att_dim: Attention dimension.
            dropout: Dropout rate.
            feat_ext: Feature extractor.
            n_outputs: Number of outputs. By default, 1 (binary classification).
            criterion: Loss function.
        """
        super(PatchGCN, self).__init__()
        self.num_outputs = n_outputs
        self.criterion = init_criterion(n_outputs, criterion)
        self.feat_ext = feat_ext

        feat_dim = get_feat_dim(feat_ext, in_shape)

        if hidden_dim is None:
            hidden_dim = feat_dim

        self.gcn_layers = torch.nn.ModuleList()
        for i in range(n_gcn_layers):
            # conv_layer = GENConv( feat_dim if i == 0 else hidden_dim, hidden_dim, aggr='softmax')
            conv_layer = GCNConv(
                feat_dim if i == 0 else hidden_dim,
                hidden_dim,
                add_self_loops=True,
                learn_weights=True,
            )
            norm_layer = torch.nn.LayerNorm(hidden_dim, elementwise_affine=True)
            act_layer = torch.nn.ReLU()
            self.gcn_layers.append(
                DeepGCNLayer(
                    conv_layer, norm_layer, act_layer, dropout=dropout, block="plain"
                )
            )

        in_mlp_dim = feat_dim + hidden_dim * (n_gcn_layers)
        self.mlp = torch.nn.ModuleList()
        for i in range(mlp_depth):
            fc_layer = LazyLinear(in_mlp_dim if i == 0 else hidden_dim, hidden_dim)
            act_layer = torch.nn.ReLU()
            dropout_layer = torch.nn.Dropout(dropout)
            self.mlp.append(torch.nn.Sequential(fc_layer, act_layer, dropout_layer))

        self.pool = AttentionPool(in_dim=hidden_dim, att_dim=att_dim)
        self.classifier = LazyLinear(hidden_dim, n_outputs)

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
        X_ = X
        for layer in self.gcn_layers:
            X = layer(X, adj)  # (batch_size, bag_size, hidden_dim)
            X_ = torch.cat([X_, X], axis=2)  # (batch_size, bag_size, hidden_dim*(i+2))
        # X_ has shape (batch_size, bag_size, hidden_dim*(n_gcn_layers+1))

        for layer in self.mlp:
            X_ = layer(X_)  # (batch_size, hidden_dim)

        if return_att:
            z, att = self.pool(X_, mask, return_att=True)  # (batch_size, hidden_dim)
        else:
            z = self.pool(X_, mask)  # (batch_size, hidden_dim)

        Y_pred = squeeze_binary_logits(self.classifier(z), self.num_outputs)

        if return_att:
            bag_size = X.shape[1]
            att = att[:, :bag_size]
            return Y_pred, att
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
        Arguments:
            Y: Bag labels of shape `(batch_size,)`.
            X: Bag features of shape `(batch_size, bag_size, ...)`.
            adj: Adjacency matrix of shape `(batch_size, bag_size, bag_size)`.
            mask: Mask of shape `(batch_size, bag_size)`.

        Returns:
            Y_pred: Bag label logits of shape `(batch_size,)`.
            loss_dict: Dictionary containing the loss
        """
        Y_pred = self.forward(X, adj, mask)
        crit_loss = compute_criterion_loss(self.criterion, Y_pred, Y, self.num_outputs)
        crit_name = self.criterion.__class__.__name__
        return Y_pred, {crit_name: crit_loss}

    def predict(
        self,
        X: torch.Tensor,
        adj: torch.Tensor,
        mask: torch.Tensor = None,
        return_inst_pred: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Arguments:
            X: Bag features of shape `(batch_size, bag_size, ...)`.
            adj: Adjacency matrix of shape `(batch_size, bag_size, bag_size)`.
            mask: Mask of shape `(batch_size, bag_size)`.
            return_inst_pred: If True, returns instance predictions.

        Returns:
            Y_pred: Bag label logits of shape `(batch_size,)`.
            y_inst_pred: If `return_inst_pred=True`, returns instance labels predictions of shape `(batch_size, bag_size)`.
        """
        return self.forward(X, adj, mask, return_att=return_inst_pred)
