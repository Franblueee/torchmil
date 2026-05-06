import torch

from .mil_model import MILModel, compute_criterion_loss, init_criterion, squeeze_binary_logits

from torchmil.nn.relprop import TransformerEncoder, Linear, IndexSelect
from torchmil.nn.utils import get_feat_dim

from torchmil.nn.gnns.dense_mincut_pool import dense_mincut_pool
from torchmil.nn.gnns.gcn_conv import GCNConv


class GTP(MILModel):
    r"""
    Method proposed in the paper [GTP: Graph-Transformer for Whole Slide Image Classification](https://arxiv.org/abs/2205.09671).

    **Forward pass.**
    Given an input bag $\mathbf{X} = \left[ \mathbf{x}_1, \ldots, \mathbf{x}_N \right]^\top \in \mathbb{R}^{N \times P}$ with adjacency matrix $\mathbf{A} \in \mathbb{R}^{N \times N}$,
    the model optionally applies a feature extractor, $\text{FeatExt}(\cdot)$, to transform the instance features: $\mathbf{X} = \text{FeatExt}(\mathbf{X}) \in \mathbb{R}^{N \times D}$.

    The bags are processed using a Graph Convolutional Network (GCN) to extract high-level instance embeddings.
    This GCN leverages a graph $\mathbf{A}$ constructed from the bag, where nodes correspond to patches, and edges are determined based on spatial adjacency:

    $$ \mathbf{H} = \text{GCN}(\mathbf{X}, \mathbf{A}) \in \mathbb{R}^{N \times D}.$$

    To reduce the number of nodes while preserving structural relationships, a min-cut pooling operation is applied:

    $$ \mathbf{X}', \mathbf{A}' = \text{MinCutPool}(\mathbf{H}, \mathbf{A}).$$

    The pooled graph is then passed through a Transformer encoder, where a class token is introduced:

    $$ \mathbf{Z} = \text{Transformer}([\text{CLS}; \mathbf{X}']) \in \mathbb{R}^{(N' + 1) \times D}.$$

    Finally, the class token representation is used for classification:

    $$ \mathbf{z} = \mathbf{Z}_{0}, \quad Y_{\text{pred}} = \text{Classifier}(\mathbf{z}).$$

    Optionally, GraphCAM can be used to generate class activation maps highlighting the most relevant regions for the classification decision.

    **Loss function.**
    By default, the model is trained end-to-end using the followind per-bag loss:

    $$ \ell = \ell_{\text{BCE}}(Y_{\text{pred}}, Y) + \ell_{\text{MinCut}}(\mathbf{X}, \mathbf{A}) + \ell_{\text{Ortho}}(\mathbf{X}, \mathbf{A}),$$

    where $\ell_{\text{BCE}}$ is the Binary Cross-Entropy loss, $\ell_{\text{MinCut}}$ is the MinCut loss, and $\ell_{\text{Ortho}}$ is the Orthogonality loss, computed during the min-cut pooling operation, see [Dense MinCut Pooling](../nn/gnns/dense_mincut_pool.md).

    """

    def __init__(
        self,
        in_shape: tuple,
        att_dim: int = 512,
        n_clusters: int = 100,
        n_layers: int = 1,
        n_heads: int = 8,
        use_mlp: bool = True,
        dropout: float = 0.0,
        feat_ext: torch.nn.Module = torch.nn.Identity(),
        n_outputs: int = 1,
        criterion: torch.nn.Module = torch.nn.BCEWithLogitsLoss(),
    ) -> None:
        """
        Class constructor.

        Arguments:
            in_shape: Shape of input data expected by the feature extractor (excluding batch dimension). If not provided, it will be lazily initialized.
            att_dim: Attention dimension for transformer encoder.
            n_clusters: Number of clusters in mincut pooling.
            n_layers: Number of layers in transformer encoder.
            n_heads: Number of heads in transformer encoder.
            use_mlp: Whether to use MLP in transformer encoder.
            dropout: Dropout rate in transformer encoder.
            feat_ext: Feature extractor.
            n_outputs: Number of outputs. By default, 1 (binary classification).
            criterion: Loss function. By default, Binary Cross-Entropy loss from logits for binary classification.
        """
        super().__init__()
        self.num_outputs = n_outputs
        self.criterion = init_criterion(n_outputs, criterion)

        self.feat_ext = feat_ext
        feat_dim = get_feat_dim(feat_ext, in_shape)

        self.gcn_conv = GCNConv(
            feat_dim,
            feat_dim,
            add_self_loops=True,
            learn_weights=True,
            activation=torch.nn.ReLU(),
        )

        self.cluster_proj = torch.nn.Linear(feat_dim, n_clusters)

        self.cls_token = torch.nn.Parameter(
            torch.zeros(1, 1, feat_dim), requires_grad=True
        )
        self.transformer_encoder = TransformerEncoder(
            in_dim=feat_dim,
            att_dim=att_dim,
            out_dim=feat_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            use_mlp=use_mlp,
            dropout=dropout,
        )
        self.classifier = Linear(feat_dim, n_outputs)
        self.index_select = IndexSelect()

    def forward(
        self,
        X: torch.Tensor,
        adj: torch.Tensor,
        mask: torch.Tensor = None,
        return_cam: bool = False,
        return_loss: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Arguments:
            X: Bag features of shape `(batch_size, bag_size, ...)`.
            adj: Adjacency matrix of shape `(batch_size, bag_size, bag_size)`.
            mask: Mask of shape `(batch_size, bag_size)`.
            return_cam: If True, returns the class activation map in addition to `Y_logits_pred`.

        Returns:
            Y_pred: Bag label logits of shape `(batch_size,)`.
            cam: Only returned when `return_cam=True`. Class activation map of shape (batch_size, bag_size).
        """

        X = self.feat_ext(X)  # (batch_size, bag_size, feat_dim)
        X = self.gcn_conv(X, adj)  # (batch_size, bag_size, feat_dim)
        S = self.cluster_proj(X)  # (batch_size, bag_size, n_clusters)

        X, adj, mc_loss, o_loss = dense_mincut_pool(
            X, adj, S, mask
        )  # X: (batch_size, n_clusters, feat_dim), adj: (batch_size, n_clusters, n_clusters), mc_loss: (batch_size, 1), o_loss: (batch_size, 1)

        cls_token = self.cls_token.repeat(X.size(0), 1, 1)
        X = torch.cat([cls_token, X], dim=1)  # (batch_size, n_clusters+1, feat_dim)

        X = self.transformer_encoder(X)  # (batch_size, n_clusters+1, feat_dim)

        # z = X[:, 0] # (batch_size, feat_dim)
        z = self.index_select(
            X, dim=1, indices=torch.tensor(0, device=X.device)
        )  # (batch_size, 1, feat_dim)

        Y_pred_raw = self.classifier(z)  # (batch_size, 1, n_outputs)

        if return_cam:
            R = self.classifier.relprop(Y_pred_raw)  # (batch_size, feat_dim)
            R = self.index_select.relprop(R)  # (batch_size, n_clusters+1, feat_dim)
            _, att_rel_list = self.transformer_encoder.relprop(
                R, return_att_relevance=True
            )  # (batch_size, n_clusters+1, feat_dim)
            cam = self._rollout_attention(
                att_rel_list
            )  # (batch_size, n_clusters+1, n_clusters+1)
            cam = cam[:, 0, 1:].unsqueeze(-1)  # (batch_size, n_clusters, 1)
            cam = torch.bmm(S, cam).squeeze(-1)  # (batch_size, bag_size)

        # Squeeze Y_pred after relprop to avoid shape mismatch
        Y_pred = squeeze_binary_logits(Y_pred_raw.squeeze(1), self.num_outputs)

        if return_loss:
            loss_dict = {"MinCutLoss": mc_loss, "OrthoLoss": o_loss}
            if return_cam:
                return Y_pred, cam, loss_dict
            else:
                return Y_pred, loss_dict
        else:
            if return_cam:
                return Y_pred, cam
            else:
                return Y_pred

    def _rollout_attention(self, att_rel_list: list[torch.Tensor]) -> torch.Tensor:
        """
        Rollout attention relevance.

        Arguments:
            att_rel_list: List of attention relevance tensors of shape `(batch_size, n_heads, n_clusters+1, n_clusters+1)`.

        Returns:
            cam: Class activation map of shape `(batch_size, n_clusters+1, n_clusters+1)`.
        """
        cam = torch.stack(
            att_rel_list, dim=0
        )  # (len, batch_size, n_heads, n_clusters+1, n_clusters+1)
        cam = cam.mean(dim=2)  # (len, batch_size, n_clusters+1, n_clusters+1)
        # add identity matrix to attention relevance
        id_mat = (
            torch.eye(cam.size(-1), device=cam.device).unsqueeze(0).unsqueeze(0)
        )  # (1, 1, n_clusters+1, n_clusters+1)
        cam = cam + id_mat
        cam = cam.prod(dim=0)  # (batch_size, n_clusters+1, n_clusters+1)
        return cam

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

        Y_pred, loss_dict = self.forward(
            X, adj, mask, return_cam=False, return_loss=True
        )

        crit_loss = compute_criterion_loss(self.criterion, Y_pred, Y, self.num_outputs)
        crit_name = self.criterion.__class__.__name__

        return Y_pred, {crit_name: crit_loss, **loss_dict}

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
            return_inst_pred: If `True`, returns instance labels predictions, in addition to bag label predictions.

        Returns:
            Y_pred: Bag label logits of shape `(batch_size,)`.
            y_inst_pred: If `return_inst_pred=True`, returns instance labels predictions of shape `(batch_size, bag_size)`.
        """
        return self.forward(X, adj, mask, return_cam=return_inst_pred)
