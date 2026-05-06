import torch

from torchmil.nn import ChebConv
from torchmil.models.mil_model import compute_criterion_loss, init_criterion, squeeze_binary_logits

from torchmil.nn.utils import get_feat_dim, LazyLinear, masked_softmax


class DeepGraphSurv(torch.nn.Module):
    r"""
    DeepGraphSurv model, as proposed in [Graph CNN for Survival Analysis on Whole Slide Pathological Images](https://link.springer.com/chapter/10.1007/978-3-030-00934-2_20).

    Given an input bag $\mathbf{X} = \left[ \mathbf{x}_1, \ldots, \mathbf{x}_N \right]^\top \in \mathbb{R}^{N \times P}$ with adjacency matrix $\mathbf{A} \in \mathbb{R}^{N \times N}$,
    the model optionally applies a feature extractor, $\text{FeatExt}(\cdot)$, to transform the instance features: $\mathbf{X} = \text{FeatExt}(\mathbf{X}) \in \mathbb{R}^{N \times D}$.

    Then, the *representation branch* transforms the instance features using a Graph Convolutional Network (GCN), and the
    *attention branch* computes the attention values $\mathbf{f}$ using another GCN,

    \begin{gather}
    \mathbf{H} = \operatorname{GCN}_{\text{rep}}(\mathbf{X}, \mathbf{A}) \in \mathbb{R}^{N \times \texttt{hidden_dim}}, \\
    \mathbf{f} = \operatorname{GCN}_{\text{att}}(\mathbf{X}, \mathbf{A}) \in \mathbb{R}^{N \times 1}.
    \end{gather}

    These GCNs are implemented using the DeepGCN layer (see [DeepGCNLayer](../nn/gnns/deepgcn.md)) with GCNConv, LayerNorm, and ReLU activation (see [GCNConv](../nn/gnns/gcn_conv.md)).

    Writing $\mathbf{H} = \left[ \mathbf{h}_1, \ldots, \mathbf{h}_N \right]^\top$,
    the attention values are used to compute the bag representation $\mathbf{z} \in \mathbb{R}^{\texttt{hidden_dim}}$ as

    \begin{equation}
    \mathbf{z} = \mathbf{H}^\top \operatorname{Softmax}(\mathbf{f}) = \sum_{n=1}^N s_n \mathbf{h}_n,
    \end{equation}

    where $s_n$ is the normalized attention score for the $n$-th instance.
    The bag representation $\mathbf{z}$ is then fed into a classifier (one linear layer) to predict the bag label.
    """

    def __init__(
        self,
        in_shape: tuple = None,
        n_layers_rep: int = 1,
        n_layers_att: int = 1,
        hidden_dim: int = None,
        att_dim: int = 128,
        dropout: float = 0.0,
        K: int = 5,
        compute_lambda_max: bool = False,
        feat_ext: torch.nn.Module = torch.nn.Identity(),
        n_outputs: int = 1,
        criterion: torch.nn.Module = torch.nn.BCEWithLogitsLoss(),
    ):
        """
        Arguments:
            in_shape: Shape of input data expected by the feature extractor (excluding batch dimension). If not provided, it will be lazily initialized.
            n_layers_rep: Number of ChebConv layers in the representation branch.
            n_layers_att: Number of ChebConv layers in the attention branch.
            hidden_dim: Hidden dimension. If not provided, it will be set to the feature dimension.
            att_dim: Attention dimension.
            dropout: Dropout rate.
            K: Order of the Chebyshev polynomial approximation for the ChebConv layers.
            compute_lambda_max: If True, computes the maximum eigenvalue of the adjacency matrix for normalization. If False, it will be set to 2.0.
            feat_ext: Feature extractor.
            n_outputs: Number of outputs. By default, 1 (binary classification).
            criterion: Loss function.
        """
        super(DeepGraphSurv, self).__init__()
        self.num_outputs = n_outputs
        self.criterion = init_criterion(n_outputs, criterion)
        self.feat_ext = feat_ext
        self.dropout_prob = dropout

        if in_shape is not None:
            feat_dim = get_feat_dim(feat_ext, in_shape)
        else:
            feat_dim = None

        if hidden_dim is None:
            hidden_dim = feat_dim

        self.layers_rep = torch.nn.ModuleList()
        for i in range(n_layers_rep):
            conv_layer = ChebConv(
                feat_dim if i == 0 else hidden_dim,
                hidden_dim,
                K=K,
                compute_lambda_max=compute_lambda_max,
            )
            self.layers_rep.append(conv_layer)

        self.layers_att = torch.nn.ModuleList()
        for i in range(n_layers_att):
            conv_layer = ChebConv(
                feat_dim if i == 0 else att_dim,
                att_dim,
                K=K,
                compute_lambda_max=compute_lambda_max,
            )
            self.layers_att.append(conv_layer)

        self.proj1d = LazyLinear(att_dim, 1)

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

        X_rep = X  # Initialize with input features
        for layer in self.layers_rep:
            X_rep = layer(X_rep, adj)
            X_rep = torch.nn.functional.relu(X_rep)
            X_rep = torch.nn.functional.dropout(
                X_rep, p=self.dropout_prob, training=self.training
            )

        X_att = X  # Initialize with input features
        for layer in self.layers_att:
            X_att = layer(X_att, adj)
            X_att = torch.nn.functional.relu(X_att)
            X_att = torch.nn.functional.dropout(
                X_att, p=self.dropout_prob, training=self.training
            )

        f = self.proj1d(X_att)  # (batch_size, bag_size, 1)
        s = masked_softmax(f, mask)  # (batch_size, bag_size, 1)

        z = torch.bmm(X_rep.transpose(1, 2), s).squeeze(-1)  # (batch_size, hidden_dim)

        Y_pred = squeeze_binary_logits(self.classifier(z), self.num_outputs)

        if return_att:
            return Y_pred, f.squeeze(-1)
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
