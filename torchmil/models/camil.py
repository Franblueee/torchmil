import torch
import torch.nn as nn

from torchmil.nn import NystromTransformerLayer
from torchmil.models import MILModel

from torchmil.nn.utils import masked_softmax, get_feat_dim


class CAMILSelfAttention(nn.Module):
    r"""
    Self-attention layer used in [CAMIL: Context-Aware Multiple Instance Learning for Cancer Detection and Subtyping in Whole Slide Images](https://arxiv.org/abs/2305.05314).
    This layer computes the self-attention values using the local information of the bag. The local information is captured using an adjacency matrix, which measures the similarity between the embeddings of instances in the bag.

    Given an input bag $\mathbf{X} = \left[ \mathbf{x}_1, \ldots, \mathbf{x}_N \right]^\top \in \mathbb{R}^{N \times P}$, and an adjacency matrix $\mathbf{A} \in \mathbb{R}^{N \times N}$, this layer computes

    $$ \mathbf{l}_i = \frac{\exp\left(\sum_{j=1}^N a_{ij} \mathbf{q}_i^\top \mathbf{k}_j \right)}{\sum_{k=1}^N \exp \left(\sum_{j=1}^N a_{kj} \mathbf{q}_k^\top \mathbf{k}_j \right)} \mathbf{v}_i,$$

    where $\mathbf{q}_i = \mathbf{W_q}\mathbf{x}_i$, $\mathbf{k}_i = \mathbf{W_k}\mathbf{x}_i$, and $\mathbf{v}_i = \mathbf{W_v}\mathbf{x}_i$ are the query, key, and value vectors, respectively.
    Finally, it returns $\mathbf{L} = \left[ \mathbf{l}_1, \ldots, \mathbf{l}_N \right]^\top$.

    """

    def __init__(self, in_dim: int, att_dim: int = 512) -> None:
        super(CAMILSelfAttention, self).__init__()
        self.qk_nn = torch.nn.Linear(in_dim, 2 * att_dim, bias=False)
        self.v_nn = torch.nn.Linear(in_dim, in_dim, bias=False)

    def forward(
        self, X: torch.Tensor, adj: torch.Tensor, mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Arguments:
            X: Bag of features of shape (batch_size, bag_size, in_dim)
            adj: Adjacency matrix of shape `(batch_size, bag_size, bag_size)`.
            mask: Mask of shape `(batch_size, bag_size)`. If None, no masking is applied.

        Returns:
            L: Self-attention vectors with shape (batch_size, bag_size, in_dim)
        """

        if mask is not None:
            mask = (
                mask[:, :, None] * mask[:, None, :]
            )  # (batch_size, bag_size, bag_size)
        else:
            mask = torch.ones(
                X.shape[0], X.shape[1], X.shape[1], device=X.device, dtype=X.dtype
            )  # (batch_size, bag_size, bag_size)

        q, k = self.qk_nn(X).chunk(
            2, dim=-1
        )  # (batch_size, bag_size, att_dim), (batch_size, bag_size, att_dim)
        v = self.v_nn(X)  # (batch_size, bag_size, in_dim)
        att_dim = q.shape[-1]

        inv_scale = 1.0 / (att_dim**0.5)

        if adj.is_sparse:
            adj = adj.to_dense()

        w = inv_scale * (q.matmul(k.transpose(-2, -1)) * adj * mask).sum(
            dim=-1, keepdim=True
        )  # (batch_size, bag_size, 1)

        L = w.softmax(dim=1) * v  # (batch_size, bag_size, in_dim)

        return L


class CAMILAttentionPool(nn.Module):
    r"""
    Attention pooling layer as described in [CAMIL: Context-Aware Multiple Instance Learning for Cancer Detection and Subtyping in Whole Slide Images](https://arxiv.org/abs/2305.05314).

    Given a bag of features $\mathbf{T} = \left[ \mathbf{t}_1, \ldots, \mathbf{t}_N \right]^\top \in \mathbb{R}^{N \times D}$ and $\mathbf{M} = \left[ \mathbf{m}_1, \ldots, \mathbf{m}_N \right]^\top \in \mathbb{R}^{N \times D}$, this layer computes the final bag representation $\mathbf{z}$ as

    \begin{gather}
    \mathbf{f} = \mathbf{w}^\top \tanh(\mathbf{T} \mathbf{W} ) \odot \operatorname{sigmoid}(\mathbf{T} \mathbf{U}), \\
    \mathbf{s} = \text{softmax}(\mathbf{f}), \\
    \mathbf{z} = \mathbf{M}^\top \mathbf{s},
    \end{gather}

    where $\mathbf{W}, \mathbf{U}$ and $\mathbf{w}$ are learnable parameters. Note the difference with conventional [AttentionPool](../nn/attention/attention_pool.md) layer, where the attention values and bag representation are computed from the same set of features.


    """

    def __init__(self, in_dim: int, att_dim: int = 128, gated: bool = False) -> None:
        super(CAMILAttentionPool, self).__init__()
        self.gated = gated
        self.fc1 = torch.nn.Linear(in_dim, att_dim)
        self.fc2 = torch.nn.Linear(att_dim, 1, bias=False)

        if self.gated:
            self.fc_gated = torch.nn.Linear(in_dim, att_dim)
            self.act_gated = torch.nn.Sigmoid()

    def forward(
        self,
        T: torch.Tensor,
        M: torch.Tensor,
        mask: torch.Tensor = None,
        return_att: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.

        Arguments:
            T: (batch_size, bag_size, in_dim)
            M: (batch_size, bag_size, in_dim)
            mask: (batch_size, bag_size)
            return_att: If True, returns attention values in addition to `z`.

        Returns:
            z: (batch_size, in_dim)
            f: (batch_size, bag_size) if `return_att
        """

        H = torch.nn.functional.tanh(self.fc1(T))  # (batch_size, bag_size, att_dim)
        if self.gated:
            G = self.act_gated(self.fc_gated(T))  # (batch_size, bag_size, att_dim)
            H = H * G

        f = self.fc2(H)  # (batch_size, bag_size, 1)
        a = masked_softmax(f, mask)  # (batch_size, bag_size, 1)
        z = torch.bmm(M.transpose(1, 2), a).squeeze(dim=2)  # (batch_size, in_dim)

        if return_att:
            return z, f.squeeze(dim=2)
        else:
            return z


class CAMIL(MILModel):
    r"""
    Context-Aware Multiple Instance Learning (CAMIL) model, presented in the paper [CAMIL: Context-Aware Multiple Instance Learning for Cancer Detection and Subtyping in Whole Slide Images](https://arxiv.org/abs/2305.05314).

    Given an input bag $\mathbf{X} = \left[ \mathbf{x}_1, \ldots, \mathbf{x}_N \right]^\top \in \mathbb{R}^{N \times P}$, the model optionally applies a feature extractor, $\text{FeatExt}(\cdot)$, to transform the instance features: $\mathbf{X} = \text{FeatExt}(\mathbf{X}) \in \mathbb{R}^{N \times D}$.

    Then, a global bag representation is computed using a [NystromTransformerLayer](../nn/transformers/nystrom_transformer.md/#torchmil.nn.transformers.NystromTransformerLayer) layer,

    $$ \mathbf{T} = \operatorname{NystromTransformerLayer}(\mathbf{X})$$

    Next, a local bag representation is computed using the [CAMILSelfAttention](./#torchmil.models.camil.CAMILSelfAttention) layer,

    $$ \mathbf{L} = \operatorname{CAMILSelfAttention}(\mathbf{T}) $$

    Finally, the local and global information is fused as

    $$ \mathbf{M} = \operatorname{sigmoid}(\mathbf{L}) \odot \mathbf{L} + (1 - \operatorname{sigmoid}(\mathbf{L})) \odot \mathbf{T},$$

    where $\odot$ denotes element-wise multiplication and $\operatorname{sigmoid}$ is the sigmoid function.

    Lastly, the final bag representation is computed using the [CAMILAttentionPool](./#torchmil.models.camil.CAMILAttentionPool), modification of the Gatted Attention Pool mechanism. The bag representation is then fed into a linear classifier to predict the bag label.
    """

    def __init__(
        self,
        in_shape: tuple,
        nystrom_att_dim: int = 512,
        pool_att_dim: int = 128,
        gated_pool: bool = False,
        n_heads: int = 4,
        n_landmarks: int = None,
        pinv_iterations: int = 6,
        dropout: float = 0.0,
        use_mlp: bool = False,
        feat_ext: torch.nn.Module = torch.nn.Identity(),
        n_outputs: int = 1,
        criterion: torch.nn.Module = torch.nn.BCEWithLogitsLoss(),
    ) -> None:
        """
        Arguments:
            in_shape: Shape of input data expected by the feature extractor (excluding batch dimension).
            pool_att_dim: Attention dimension for the attention pooling layer.
            gated_pool: If True, use gated attention pooling.
            nystrom_att_dim: Attention dimension for the Nystrom Transformer layer.
            n_heads: Number of attention heads in the Nystrom Transformer layer.
            n_landmarks: Number of landmarks in the Nystrom Transformer layer.
            pinv_iterations: Number of iterations for computing the pseudo-inverse in the Nystrom Transformer layer.
            dropout: Dropout rate of the Nystrom Transformer Layer.
            use_mlp: If True, use MLP in the Nystrom Transformer layer.
            feat_ext: Feature extractor.
            n_outputs: Number of outputs. By default, 1 (binary classification).
            criterion: Loss function. By default, Binary Cross-Entropy loss from logits.
        """
        super(CAMIL, self).__init__()
        self.num_outputs = n_outputs
        self.feat_ext = feat_ext
        self.criterion = criterion

        feat_dim = get_feat_dim(feat_ext, in_shape)

        if feat_dim != nystrom_att_dim:
            self.fc1 = torch.nn.Linear(feat_dim, nystrom_att_dim)
        else:
            self.fc1 = torch.nn.Identity()

        self.nystrom_transformer_layer = NystromTransformerLayer(
            in_dim=nystrom_att_dim,
            att_dim=nystrom_att_dim,
            n_heads=n_heads,
            n_landmarks=n_landmarks,
            pinv_iterations=pinv_iterations,
            dropout=dropout,
            use_mlp=use_mlp,
        )

        self.camil_self_attention = CAMILSelfAttention(
            in_dim=nystrom_att_dim, att_dim=nystrom_att_dim
        )
        self.camil_att_pool = CAMILAttentionPool(
            in_dim=nystrom_att_dim, att_dim=pool_att_dim, gated=gated_pool
        )

        self.classifier = nn.Linear(nystrom_att_dim, n_outputs)

        self.criterion = criterion

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
        T = self.nystrom_transformer_layer(
            self.fc1(X)
        )  # (batch_size, bag_size, feat_dim)

        L = self.camil_self_attention(T, adj, mask)  # (batch_size, bag_size, feat_dim)

        M = (
            torch.sigmoid(L) * L + (1 - torch.sigmoid(L)) * T
        )  # (batch_size, bag_size, feat_dim)

        if return_att:
            z, att = self.camil_att_pool(
                T, M, mask, return_att=True
            )  # (batch_size, feat_dim), (batch_size, bag_size)
        else:
            z = self.camil_att_pool(T, M, mask)  # (batch_size, feat_dim)

        Y_pred = self.classifier(z)  # (batch_size, n_outputs)
        if self.num_outputs == 1:
            Y_pred = Y_pred.squeeze(dim=-1)  # (batch_size,)

        if return_att:
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

        crit_loss = self.criterion(Y_pred.float(), Y.float())
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
