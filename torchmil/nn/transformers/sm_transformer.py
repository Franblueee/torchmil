import torch
from torch.nn.attention import SDPBackend

from torchmil.nn import MultiheadSelfAttention, Sm

from .encoder import Encoder
from .layer import Layer


SDP_BACKEND = [
    SDPBackend.MATH,
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION,
    SDPBackend.CUDNN_ATTENTION,
]


class SmMultiheadSelfAttention(MultiheadSelfAttention):
    r"""
    Multihead self-attention layer with the $\texttt{Sm}$ operator.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = None,
        att_dim: int = 512,
        n_heads: int = 4,
        dropout: float = 0.0,
        sm_alpha: float = "trainable",
        sm_mode: str = "approx",
        sm_steps: int = 10,
    ):
        """
        Arguments:
            in_dim: Input dimension.
            out_dim: Output dimension. If None, out_dim = in_dim.
            att_dim: Attention dimension.
            n_heads: Number of heads.
            dropout: Dropout rate.
            sm_alpha: Alpha value for the Sm operator.
            sm_mode: Sm mode ('exact' or 'approx').
            sm_steps: Number of steps to approximate the exact Sm operator.
        """

        super(SmMultiheadSelfAttention, self).__init__(
            att_dim=att_dim,
            in_dim=in_dim,
            out_dim=out_dim,
            n_heads=n_heads,
            dropout=dropout,
        )

        self.sm = Sm(alpha=sm_alpha, mode=sm_mode, num_steps=sm_steps)

    def forward(
        self,
        X: torch.Tensor,
        adj: torch.Tensor,
        mask: torch.Tensor = None,
        return_att: bool = False,
    ) -> torch.Tensor:
        """
        Forward method.

        Arguments:
            X: Input tensor of shape `(batch_size, bag_size, in_dim)`.
            adj: Adjacency matrix of shape `(batch_size, bag_size, bag_size)`.
            mask: Mask tensor of shape `(batch_size, bag_size)`.
            return_att: If True, returns attention weights, of shape `(batch_size, n_heads, bag_size, bag_size)`.
        """
        out = super().forward(X, mask=mask, return_att=return_att)
        if return_att:
            Y, att = out
        else:
            Y = out
        Y = self.sm(Y, adj)
        if return_att:
            return Y, att
        else:
            return Y


class SmTransformerLayer(Layer):
    r"""
    One layer of the Transformer encoder with the $\texttt{Sm}$ operator.

    Given an input bag $\mathbf{X} = \left[ \mathbf{x}_1, \ldots, \mathbf{x}_N \right]^\top \in \mathbb{R}^{N \times D}$,
    this module computes:

    \begin{align*}
    \mathbf{Z} & = \mathbf{X} + \texttt{Sm}( \text{SelfAttention}( \text{LayerNorm}(\mathbf{X}) ) )\\
    \mathbf{Y} & = \mathbf{Z} + \text{MLP}(\text{LayerNorm}(\mathbf{Z})), \\
    \end{align*}

    and outputs $\mathbf{Y}$.

    See [Sm](../sm.md) for more details on the Sm operator.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = None,
        att_dim: int = 512,
        n_heads: int = 4,
        use_mlp: bool = True,
        dropout: float = 0.0,
        sm_alpha: float = "trainable",
        sm_mode: str = "approx",
        sm_steps: int = 10,
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
            sm_alpha: Alpha value for the Sm operator.
            sm_mode: Sm mode.
            sm_steps: Number of steps to approximate the exact Sm operator.
        """

        att_module = SmMultiheadSelfAttention(
            att_dim=att_dim,
            in_dim=in_dim,
            out_dim=att_dim,
            n_heads=n_heads,
            dropout=dropout,
            sm_alpha=sm_alpha,
            sm_mode=sm_mode,
            sm_steps=sm_steps,
        )

        super(SmTransformerLayer, self).__init__(
            in_dim=in_dim,
            att_in_dim=in_dim,
            out_dim=out_dim,
            att_out_dim=att_dim,
            att_module=att_module,
            use_mlp=use_mlp,
            dropout=dropout,
        )

    def forward(
        self,
        X: torch.Tensor,
        adj: torch.Tensor,
        mask: torch.Tensor = None,
        return_att: bool = False,
    ) -> torch.Tensor:
        """
        Forward method.

        Arguments:
            X: Input tensor of shape `(batch_size, bag_size, in_dim)`.
            adj: Adjacency matrix of shape `(batch_size, bag_size, bag_size)`.
            mask: Mask tensor of shape `(batch_size, bag_size)`.
            return_att: If True, returns attention weights, of shape `(batch_size, n_heads, bag_size, bag_size)`.

        Returns:
            Y: Output tensor of shape `(batch_size, bag_size, in_dim)`.
        """

        return super().forward(X, mask=mask, adj=adj, return_att=return_att)


class SmTransformerEncoder(Encoder):
    r"""
    A Transformer encoder with the $\texttt{Sm}$ operator, skip connections and layer normalization.

    Given an input bag input bag $\mathbf{X} = \left[ \mathbf{x}_1, \ldots, \mathbf{x}_N \right]^\top \in \mathbb{R}^{N \times D}$,
    it computes:

    \begin{align*}
    \mathbf{X}^{0} & = \mathbf{X} \\
    \mathbf{Z}^{l} & = \mathbf{X}^{l-1} + \texttt{Sm}( \text{SelfAttention}( \text{LayerNorm}(\mathbf{X}^{l-1}) ) ), \quad l = 1, \ldots, L \\
    \mathbf{X}^{l} & = \mathbf{Z}^{l} + \text{MLP}(\text{LayerNorm}(\mathbf{Z}^{l})), \quad l = 1, \ldots, L \\
    \end{align*}

    This module outputs $\text{SmTransformerEncoder}(\mathbf{X}) = \mathbf{X}^{L}$ if `add_self=False`,
    and $\text{SmTransformerEncoder}(\mathbf{X}) = \mathbf{X}^{L} + \mathbf{X}$ if `add_self=True`.

    See [Sm](../sm.md) for more details on the Sm operator.
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
        sm_alpha: float = "trainable",
        sm_mode: str = "approx",
        sm_steps: int = 10,
    ):
        """
        Class constructor

        Arguments:
            in_dim: Input dimension.
            out_dim: Output dimension. If None, out_dim = in_dim.
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
                SmTransformerLayer(
                    in_dim=in_dim if i == 0 else att_dim,
                    out_dim=out_dim if i == n_layers - 1 else att_dim,
                    att_dim=att_dim,
                    n_heads=n_heads,
                    use_mlp=use_mlp,
                    dropout=dropout,
                    sm_alpha=sm_alpha,
                    sm_mode=sm_mode,
                    sm_steps=sm_steps,
                )
                for i in range(n_layers)
            ]
        )

        super(SmTransformerEncoder, self).__init__(layers, add_self=add_self)

        self.norm = torch.nn.LayerNorm(out_dim)

    def forward(
        self,
        X: torch.Tensor,
        adj: torch.Tensor,
        mask: torch.Tensor = None,
        return_att: bool = False,
    ) -> torch.Tensor:
        """
        Forward method.

        Arguments:
            X: Input tensor of shape `(batch_size, bag_size, in_dim)`.
            adj: Adjacency matrix of shape `(batch_size, bag_size, bag_size)`.
            mask: Mask tensor of shape `(batch_size, bag_size)`.

        Returns:
            Y: Output tensor of shape `(batch_size, bag_size, in_dim)`.
        """

        out = super().forward(X, mask=mask, adj=adj, return_att=return_att)

        if return_att:
            Y, att = out
            Y = self.norm(Y)  # (batch_size, bag_size, att_dim)
            return Y, att
        else:
            Y = out
            Y = self.norm(Y)  # (batch_size, bag_size, att_dim)
            return Y
