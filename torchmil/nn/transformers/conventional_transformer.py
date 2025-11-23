import torch
from torch.nn.attention import SDPBackend

from torchmil.nn import MultiheadSelfAttention

from .encoder import Encoder
from .layer import Layer


SDP_BACKEND = [
    SDPBackend.MATH,
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION,
    SDPBackend.CUDNN_ATTENTION,
]


class TransformerLayer(Layer):
    r"""
    One layer of the Transformer encoder.

    Given an input bag $\mathbf{X} = \left[ \mathbf{x}_1, \ldots, \mathbf{x}_N \right]^\top \in \mathbb{R}^{N \times D}$,
    this module computes:

    \begin{align*}
    \mathbf{Z} & = \mathbf{X} + \operatorname{SelfAttention}( \operatorname{LayerNorm}(\mathbf{X}) ) \\
    \mathbf{Y} & = \mathbf{Z} + \operatorname{MLP}(\operatorname{LayerNorm}(\mathbf{Z})), \\
    \end{align*}

    and outputs $\mathbf{Y}$.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = None,
        att_dim: int = 512,
        n_heads: int = 4,
        use_mlp: bool = True,
        dropout: float = 0.0,
    ):
        """
        Class constructor.

        Arguments:
            in_dim: Input dimension.
            out_dim: Output dimension. If None, out_dim = in_dim.
            att_dim: Attention dimension.
            n_heads: Number of heads.
            use_mlp: Whether to use feedforward layer.
            dropout: Dropout rate
        """

        att_module = MultiheadSelfAttention(
            att_dim=att_dim,
            in_dim=in_dim,
            out_dim=att_dim,
            n_heads=n_heads,
            dropout=dropout,
        )

        super(TransformerLayer, self).__init__(
            in_dim=in_dim,
            att_in_dim=in_dim,
            out_dim=out_dim,
            att_out_dim=att_dim,
            att_module=att_module,
            use_mlp=use_mlp,
            dropout=dropout,
        )

    def forward(
        self, X: torch.Tensor, mask: torch.Tensor = None, return_att: bool = False
    ) -> torch.Tensor:
        """
        Forward method.

        Arguments:
            X: Input tensor of shape `(batch_size, bag_size, in_dim)`.
            mask: Mask tensor of shape `(batch_size, bag_size)`.
            return_att: If True, returns attention weights, of shape `(batch_size, n_heads, bag_size, bag_size)`.

        Returns:
            Y: Output tensor of shape `(batch_size, bag_size, out_dim)`.
        """

        return super().forward(X, mask=mask, return_att=return_att)


class TransformerEncoder(Encoder):
    r"""
    A Transformer encoder with skip connections and layer normalization.

    Given an input bag input bag $\mathbf{X} = \left[ \mathbf{x}_1, \ldots, \mathbf{x}_N \right]^\top \in \mathbb{R}^{N \times D}$,
    it computes:

    \begin{align*}
    \mathbf{X}^{0} & = \mathbf{X} \\
    \mathbf{Z}^{l} & = \mathbf{X}^{l-1} + \operatorname{SelfAttention}( \operatorname{LayerNorm}(\mathbf{X}^{l-1}) ), \quad l = 1, \ldots, L \\
    \mathbf{X}^{l} & = \mathbf{Z}^{l} + \operatorname{MLP}(\operatorname{LayerNorm}(\mathbf{Z}^{l})), \quad l = 1, \ldots, L. \\
    \end{align*}

    This module outputs $\operatorname{TransformerEncoder}(\mathbf{X}) = \mathbf{X}^{L}$ if `add_self=False`,
    and $\operatorname{TransformerEncoder}(\mathbf{X}) = \mathbf{X}^{L} + \mathbf{X}$ if `add_self=True`.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = None,
        att_dim: int = 512,
        n_heads: int = 4,
        n_layers: int = 4,
        use_mlp: bool = True,
        add_self: bool = False,
        dropout: float = 0.0,
    ):
        """
        Class constructor

        Arguments:
            in_dim: Input dimension.
            out_dim: Output dimension. If None, `out_dim = in_dim`.
            att_dim: Attention dimension.
            n_heads: Number of heads.
            n_layers: Number of layers.
            use_mlp: Whether to use feedforward layer.
            add_self: Whether to add input to output. If True, `att_dim` must be equal to `in_dim`.
            dropout: Dropout rate.
        """

        if out_dim is None:
            out_dim = in_dim

        layers = torch.nn.ModuleList(
            [
                TransformerLayer(
                    in_dim=in_dim if i == 0 else att_dim,
                    out_dim=out_dim if i == n_layers - 1 else att_dim,
                    att_dim=att_dim,
                    n_heads=n_heads,
                    use_mlp=use_mlp,
                    dropout=dropout,
                )
                for i in range(n_layers)
            ]
        )

        super(TransformerEncoder, self).__init__(layers, add_self=add_self)

        self.norm = torch.nn.LayerNorm(out_dim)

    def forward(
        self, X: torch.Tensor, mask: torch.Tensor = None, return_att: bool = False
    ) -> torch.Tensor:
        """
        Forward method.

        Arguments:
            X: Input tensor of shape `(batch_size, bag_size, in_dim)`.
            mask: Mask tensor of shape `(batch_size, bag_size)`.
            return_att: If True, returns attention weights, of shape `(n_layers, batch_size, n_heads, bag_size, bag_size)`.

        Returns:
            Y: Output tensor of shape `(batch_size, bag_size, in_dim)`.
        """

        if return_att:
            Y, att = super().forward(X, mask=mask, return_att=True)
            Y = self.norm(Y)  # (batch_size, bag_size, att_dim)
            return Y, att
        else:
            Y = super().forward(X, mask=mask, return_att=False)
            Y = self.norm(Y)  # (batch_size, bag_size, att_dim)
            return Y
