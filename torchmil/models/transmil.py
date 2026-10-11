import math

import torch

from torchmil.nn.utils import get_feat_dim
from torchmil.nn import NystromTransformerLayer
from torchmil.models import MILModel


class PPEG(torch.nn.Module):
    """
    Pyramid Positional Encoding Generator, as described in the [TransMIL paper](https://arxiv.org/abs/2106.00908).

    """

    def __init__(self, dim: int) -> None:
        """
        Arguments:
            dim: Input dimension.
        """
        super(PPEG, self).__init__()
        self.proj = torch.nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = torch.nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = torch.nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Forward pass.

        Arguments:
            x: Input tensor of shape `(batch_size, H*W+1, dim)`.
            H: Height of the grid.
            W: Width of the grid.

        Returns:
            y: Output tensor of shape `(batch_size, H*W+1, dim)`.
        """
        batch_size, _, dim = x.shape
        cls_token, feat_token = (
            x[:, 0],
            x[:, 1:],
        )  # (batch_size, dim), (batch_size, H*W, dim)
        cnn_feat = feat_token.transpose(1, 2).view(
            batch_size, dim, H, W
        )  # (batch_size, dim, H, W)
        cnn_proj = self.proj(cnn_feat)
        cnn_proj1 = self.proj1(cnn_feat)
        cnn_proj2 = self.proj2(cnn_feat)
        y = cnn_feat + cnn_proj + cnn_proj1 + cnn_proj2  # (batch_size, dim, H, W)
        y = y.flatten(2).transpose(1, 2)  # (batch_size, H*W, dim)
        y = torch.cat((cls_token.unsqueeze(1), y), dim=1)  # (batch_size, H*W+1, dim)
        return y


class TransMIL(MILModel):
    r"""
    Method proposed in the paper [TransMIL: Transformer based Correlated Multiple Instance Learning for Whole Slide Image Classification](https://arxiv.org/abs/2106.00908).

    Given an input bag $\mathbf{X} = \left[ \mathbf{x}_1, \ldots, \mathbf{x}_N \right]^\top \in \mathbb{R}^{N \times P}$,
    the model optionally applies a feature extractor, $\text{FeatExt}(\cdot)$, to transform the instance features: $\mathbf{X} = \text{FeatExt}(\mathbf{X}) \in \mathbb{R}^{N \times D}$.

    Then, following Algorithm 2 in the paper, it performs sequence squaring, adds a class token, and applies the novel TPT module. This module consists of two [Nyströmformer](https://arxiv.org/abs/2102.03902) layers and the novel PPEG (Pyramid Positional Encoding Generator) layer.

    Finally, a linear classifier is used to predict the bag label from the class token.
    """

    def __init__(
        self,
        in_shape: tuple,
        att_dim: int = 512,
        n_layers: int = 2,
        n_heads: int = 4,
        n_landmarks: int = None,
        pinv_iterations: int = 6,
        dropout: float = 0.0,
        use_mlp: bool = False,
        feat_ext: torch.nn.Module = torch.nn.Identity(),
        criterion: torch.nn.Module = torch.nn.BCEWithLogitsLoss(),
    ):
        r"""
        Arguments:
            in_shape: Shape of input data expected by the feature extractor (excluding batch dimension).
            att_dim: Embedding dimension. Should be divisible by `n_heads`.
            n_layers: Number of Nyströmformer layers.
            n_heads: Number of heads in the Nyströmformer layer.
            n_landmarks: Number of landmarks in the Nyströmformer layer.
            pinv_iterations: Number of iterations for the pseudo-inverse in the Nyströmformer layer.
            dropout: Dropout rate in the Nyströmformer layer.
            use_mlp: Whether to use a MLP after the Nyströmformer layer.
            feat_ext: Feature extractor. By default, the identity function (no feature extraction).
            criterion: Loss function. By default, Binary Cross-Entropy loss from logits.
        """

        assert n_layers >= 2, "Number of layers must be at least 2."

        super(TransMIL, self).__init__()

        if n_landmarks is None:
            n_landmarks = att_dim // 2
        self.n_landmarks = n_landmarks

        self.feat_ext = feat_ext

        feat_dim = get_feat_dim(feat_ext, in_shape)

        if feat_dim != att_dim:
            self.fc1 = torch.nn.Linear(feat_dim, att_dim)
        else:
            self.fc1 = torch.nn.Identity()

        self.pos_layer = PPEG(dim=att_dim)
        self.cls_token = torch.nn.Parameter(torch.randn(1, 1, att_dim))

        self.layers = torch.nn.ModuleList(
            [
                NystromTransformerLayer(
                    in_dim=att_dim,
                    att_dim=att_dim,
                    out_dim=att_dim,
                    n_heads=n_heads,
                    n_landmarks=n_landmarks,
                    pinv_iterations=pinv_iterations,
                    dropout=dropout,
                    use_mlp=use_mlp,
                    learn_weights=True,
                )
                for _ in range(n_layers)
            ]
        )

        self.norm = torch.nn.LayerNorm(att_dim)
        self.classifier = torch.nn.Linear(att_dim, 1)

        self.criterion = criterion

    def _sequence_squaring(
        self, X: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Arrange the instances of each bag in a square grid, row by row.

        The side of the grid of a bag is the smallest one that fits its instances, and the cells left
        after the last instance repeat the first instances of the bag. The grids of all the bags are
        returned in the top-left corner of a grid as large as the largest one, with the rest of the cells empty.

        Arguments:
            X: Input tensor of shape `(batch_size, bag_size, att_dim)`.
            mask: Mask of shape `(batch_size, bag_size)`.

        Returns:
            X: Grid of instances of shape `(batch_size, grid_size*grid_size, att_dim)`, with zeros in the empty cells.
            grid_mask: Non-empty cells of the grid, of shape `(batch_size, grid_size*grid_size)`.
            side: Side of the grid of each bag, of shape `(batch_size,)`.
            grid_size: Side of the returned grid.
        """
        device = X.device
        att_dim = X.shape[2]

        n_instances = mask.sum(dim=1)  # (batch_size,)
        if (n_instances == 0).any():
            raise ValueError("All bags must have at least one instance.")
        # smallest side whose square fits the instances
        side = torch.tensor(
            [math.isqrt(n - 1) + 1 for n in n_instances.tolist()], device=device
        )  # (batch_size,)
        grid_size = int(side.max())

        cell = torch.arange(grid_size * grid_size, device=device)
        row, col = cell // grid_size, cell % grid_size  # (grid_size*grid_size,)
        grid_mask = (row[None, :] < side[:, None]) & (
            col[None, :] < side[:, None]
        )  # (batch_size, grid_size*grid_size)

        # position of each cell in the squared sequence of its bag, and instance it holds
        seq_pos = row[None, :] * side[:, None] + col[None, :]
        instance = seq_pos % n_instances[:, None]  # (batch_size, grid_size*grid_size)

        # the i-th instance of a bag is its i-th non-masked position
        order = torch.argsort(mask.int(), dim=1, descending=True, stable=True)
        index = torch.gather(order, 1, instance)  # (batch_size, grid_size*grid_size)

        X = torch.gather(
            X, 1, index[..., None].expand(-1, -1, att_dim)
        )  # (batch_size, grid_size*grid_size, att_dim)
        X = X * grid_mask[..., None]

        return X, grid_mask, side, grid_size

    def forward(
        self, X: torch.Tensor, mask: torch.Tensor = None, return_att: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Arguments:
            X: Input tensor of shape `(batch_size, bag_size, in_dim)`.
            mask: Mask of shape `(batch_size, bag_size)`.
            return_att: Whether to return the attention values.

        Returns:
            Y_pred: Bag label logits of shape `(batch_size,)`.
            att: Only returned when `return_att=True`. Attention values of shape (batch_size, bag_size).
        """

        device = X.device
        batch_size, bag_size = X.shape[0], X.shape[1]

        X = self.feat_ext(X)  # (batch_size, bag_size, feat_dim)
        X = self.fc1(X)  # (batch_size, bag_size, att_dim)

        if mask is None:
            mask = torch.ones(batch_size, bag_size, device=device)
        mask = mask.bool()

        # pad
        X, grid_mask, side, grid_size = self._sequence_squaring(
            X, mask
        )  # (batch_size, grid_size*grid_size, att_dim), (batch_size, grid_size*grid_size)

        # add cls_token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1).to(
            device
        )  # (batch_size, 1, att_dim)
        X = torch.cat(
            (cls_tokens, X), dim=1
        )  # (batch_size, grid_size*grid_size+1, att_dim)
        seq_mask = torch.nn.functional.pad(
            grid_mask, (1, 0), value=True
        )  # (batch_size, grid_size*grid_size+1)

        # first transformer layer
        X = self.layers[0](
            X, mask=seq_mask
        )  # (batch_size, grid_size*grid_size+1, att_dim)

        # pos layer
        # the empty cells must be zero, like the border the convolutions add around the grid
        X = X * seq_mask[..., None]
        X = self.pos_layer(
            X, grid_size, grid_size
        )  # (batch_size, grid_size*grid_size+1, att_dim)

        # remaining transformer layers (except the last one)
        for layer in self.layers[1:-1]:
            X = layer(X, mask=seq_mask)

        # last transformer layer
        if return_att:
            X, attn = self.layers[-1](
                X, mask=seq_mask, return_att=True
            )  # (batch_size, grid_size*grid_size+1, att_dim), (batch_size, n_heads, grid_size*grid_size+1, grid_size*grid_size+1)
            attn = attn.mean(
                dim=1
            )  # (batch_size, grid_size*grid_size+1, grid_size*grid_size+1)
            attn = attn[:, 0, 1:]  # (batch_size, grid_size*grid_size)
            # remove padding: keep the cell where each instance was first placed
            instance = (mask.cumsum(dim=1) - 1).clamp(min=0)  # (batch_size, bag_size)
            cell = (instance // side[:, None]) * grid_size + instance % side[:, None]
            attn = torch.gather(attn, 1, cell) * mask  # (batch_size, bag_size)
        else:
            X = self.layers[-1](
                X, mask=seq_mask
            )  # (batch_size, grid_size*grid_size+1, att_dim)

        # norm layer
        X = self.norm(X)  # (batch_size, grid_size*grid_size+1, att_dim)

        # cls_token
        cls_token = X[:, 0]  # (batch_size, att_dim)

        # predict
        bag_pred = self.classifier(cls_token).squeeze(-1)  # (batch_size,)

        if return_att:
            return bag_pred, attn
        else:
            return bag_pred

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
            X: Input tensor of shape `(batch_size, bag_size, in_dim)`.
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
            X: Input tensor of shape `(batch_size, bag_size, in_dim)`.
            mask: Mask of shape `(batch_size, bag_size)`.
            return_inst_pred: If `True`, returns instance labels predictions, in addition to bag label predictions.

        Returns:
            Y_pred: Bag label logits of shape `(batch_size,)`.
            att: Only returned when `return_att=True`. Attention values of shape (batch_size, bag_size).
        """
        return self.forward(X, mask, return_att=return_inst_pred)
