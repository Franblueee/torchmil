import torch
import numpy as np

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
        n_outputs: int = 1,
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
            n_outputs: Number of outputs. By default, 1 (binary classification).
            criterion: Loss function. By default, Binary Cross-Entropy loss from logits.
        """

        assert n_layers >= 2, "Number of layers must be at least 2."

        super(TransMIL, self).__init__()

        if n_landmarks is None:
            n_landmarks = att_dim // 2
        self.n_landmarks = n_landmarks
        self.num_outputs = n_outputs

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
        self.classifier = torch.nn.Linear(att_dim, n_outputs)

        self.criterion = criterion

    def forward(
        self, X: torch.Tensor, return_att: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Arguments:
            X: Input tensor of shape `(batch_size, bag_size, in_dim)`.
            return_att: Whether to return the attention values.

        Returns:
            Y_pred: Bag label logits of shape `(batch_size,)`.
            att: Only returned when `return_att=True`. Attention values of shape (batch_size, bag_size).
        """

        device = X.device
        batch_size, bag_size = X.shape[0], X.shape[1]

        X = self.feat_ext(X)  # (batch_size, bag_size, feat_dim)
        X = self.fc1(X)  # (batch_size, bag_size, att_dim)

        # pad
        bag_size = X.shape[1]
        padded_size = int(np.ceil(np.sqrt(bag_size)))
        add_length = padded_size * padded_size - bag_size
        X = torch.cat(
            [X, X[:, :add_length, :]], dim=1
        )  # (batch_size, padded_size*padded_size, att_dim)

        # add cls_token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1).to(
            device
        )  # (batch_size, 1, att_dim)
        X = torch.cat(
            (cls_tokens, X), dim=1
        )  # (batch_size, padded_size*padded_size+1, att_dim)

        # first transformer layer
        X = self.layers[0](X)  # (batch_size, padded_size*padded_size+1, att_dim)

        # pos layer
        X = self.pos_layer(
            X, padded_size, padded_size
        )  # (batch_size, padded_size*padded_size+1, att_dim)

        # remaining transformer layers (except the last one)
        for layer in self.layers[1:-1]:
            X = layer(X)

        # last transformer layer
        if return_att:
            X, attn = self.layers[-1](
                X, return_att=True
            )  # (batch_size, padded_size*padded_size+1, att_dim), (batch_size, n_heads, padded_size*padded_size+1, padded_size*padded_size+1)
            # remove padding
            attn = attn.mean(
                dim=1
            )  # (batch_size, padded_size*padded_size+1, padded_size*padded_size+1)
            attn = attn[:, 0, 1 : bag_size + 1]  # (batch_size, padded_size*padded_size)
        else:
            X = self.layers[-1](X)  # (batch_size, padded_size*padded_size+1, att_dim)

        # norm layer
        X = self.norm(X)  # (batch_size, padded_size*padded_size+1, att_dim)

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
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute loss given true bag labels.

        Arguments:
            Y: Bag labels of shape `(batch_size,)`.
            X: Input tensor of shape `(batch_size, bag_size, in_dim)`.

        Returns:
            Y_pred: Bag label logits of shape `(batch_size,)`.
            loss_dict: Dictionary containing the loss value.
        """

        Y_pred = self.forward(X, return_att=False)

        crit_loss = self.criterion(Y_pred.float(), Y.float())
        crit_name = self.criterion.__class__.__name__

        return Y_pred, {crit_name: crit_loss}

    def predict(
        self, X: torch.Tensor, return_inst_pred: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict bag and (optionally) instance labels.

        Arguments:
            X: Input tensor of shape `(batch_size, bag_size, in_dim)`.
            return_inst_pred: If `True`, returns instance labels predictions, in addition to bag label predictions.

        Returns:
            Y_pred: Bag label logits of shape `(batch_size,)`.
            att: Only returned when `return_att=True`. Attention values of shape (batch_size, bag_size).
        """
        return self.forward(X, return_att=return_inst_pred)
