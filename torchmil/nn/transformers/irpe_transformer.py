import torch
from torch.nn.attention import SDPBackend

from torchmil.nn import iRPEMultiheadSelfAttention

from .encoder import Encoder
from .layer import Layer


SDP_BACKEND = [
    SDPBackend.MATH,
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION,
    SDPBackend.CUDNN_ATTENTION,
]


class iRPETransformerLayer(Layer):
    r"""
    Transformer layer with image Relative Position Encoding (iRPE), as described in [Rethinking and Improving Relative Position Encoding for Vision Transformer](https://openaccess.thecvf.com/content/ICCV2021/html/Wu_Rethinking_and_Improving_Relative_Position_Encoding_for_Vision_Transformer_ICCV_2021_paper.html).

    Given an input bag $\mathbf{X} = \left[ \mathbf{x}_1, \ldots, \mathbf{x}_N \right]^\top \in \mathbb{R}^{N \times D}$,
    this module computes:

    \begin{align*}
    \mathbf{Z} & = \mathbf{X} + \operatorname{iRPESelfAttention}( \operatorname{LayerNorm}(\mathbf{X}) ) \\
    \mathbf{Y} & = \mathbf{Z} + \operatorname{MLP}(\operatorname{LayerNorm}(\mathbf{Z})), \\
    \end{align*}

    and outputs $\mathbf{Y}$. See [iRPEMultiheadSelfAttention](../attention/irpe_multihead_self_attention.md) for more details about $\operatorname{iRPESelfAttention}$.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = None,
        att_dim: int = 512,
        n_heads: int = 4,
        use_mlp: bool = True,
        dropout: float = 0.0,
        rpe_ratio: float = 1.9,
        rpe_method: str = "product",
        rpe_mode: str = "contextual",
        rpe_shared_head: bool = True,
        rpe_skip: int = 1,
        rpe_on: str = "k",
    ):
        """
        Class constructor.

        Arguments:
            att_dim: Attention dimension.
            in_dim: Input dimension. If None, in_dim = att_dim.
            out_dim: Output dimension. If None, out_dim = in_dim.
            n_heads: Number of heads.
            use_mlp: Whether to use feedforward layer.
            dropout: Dropout rate.
            rpe_ratio: Relative position encoding ratio.
            rpe_method: Relative position encoding method. Possible values: ['euc', 'quant', 'cross', 'product']
            rpe_mode: Relative position encoding mode. Possible values: [None, 'bias', 'contextual']
            rpe_shared_head: Whether to share weights across heads.
            rpe_skip: Relative position encoding skip. Possible values: [0, 1].
            rpe_on: Where to apply relative positional encoding. Possible values: ['q', 'k', 'v', 'qk', 'kv', 'qkv'].
        """

        att_module = iRPEMultiheadSelfAttention(
            att_dim=att_dim,
            in_dim=in_dim,
            out_dim=att_dim,
            n_heads=n_heads,
            dropout=dropout,
            rpe_ratio=rpe_ratio,
            rpe_method=rpe_method,
            rpe_mode=rpe_mode,
            rpe_shared_head=rpe_shared_head,
            rpe_skip=rpe_skip,
            rpe_on=rpe_on,
        )

        super(iRPETransformerLayer, self).__init__(
            in_dim=in_dim,
            att_in_dim=in_dim,
            out_dim=out_dim,
            att_out_dim=att_dim,
            att_module=att_module,
            use_mlp=use_mlp,
            dropout=dropout,
        )

    def forward(self, X: torch.Tensor, return_att: bool = False) -> torch.Tensor:
        """
        Forward method.

        Arguments:
            X: Input tensor of shape `(batch_size, bag_size, in_dim)`.

        Returns:
            Y: Output tensor of shape `(batch_size, bag_size, out_dim)`.
        """

        return super().forward(X, return_att=return_att)


class iRPETransformerEncoder(Encoder):
    r"""
    Given an input bag input bag $\mathbf{X} = \left[ \mathbf{x}_1, \ldots, \mathbf{x}_N \right]^\top \in \mathbb{R}^{N \times D}$,
    it computes:

    \begin{align*}
    \mathbf{X}^{0} & = \mathbf{X} \\
    \mathbf{Z}^{l} & = \mathbf{X}^{l-1} + \operatorname{iRPESelfAttention}( \operatorname{LayerNorm}(\mathbf{X}^{l-1}) ), \quad l = 1, \ldots, L \\
    \mathbf{X}^{l} & = \mathbf{Z}^{l} + \operatorname{MLP}(\operatorname{LayerNorm}(\mathbf{Z}^{l})), \quad l = 1, \ldots, L. \\
    \end{align*}

    See [iRPEMultiheadSelfAttention](../attention/irpe_multihead_self_attention.md) for more details about $\operatorname{iRPESelfAttention}$.

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
        rpe_ratio: float = 1.9,
        rpe_method: str = "product",
        rpe_mode: str = "contextual",
        rpe_shared_head: bool = True,
        rpe_skip: int = 1,
        rpe_on: str = "k",
    ):
        """
        Class constructor

        Arguments:
            in_dim: Input dimension.
            att_dim: Attention dimension.
            out_dim: Output dimension. If None, out_dim = in_dim.
            n_heads: Number of heads.
            n_layers: Number of layers.
            use_mlp: Whether to use feedforward layer.
            add_self: Whether to add input to output.
            dropout: Dropout rate.
            rpe_ratio: Relative position encoding ratio.
            rpe_method: Relative position encoding method. Possible values: ['euc', 'quant', 'cross', 'product']
            rpe_mode: Relative position encoding mode. Possible values: [None, 'bias', 'contextual']
            rpe_shared_head: Whether to share weights across heads.
            rpe_skip: Relative position encoding skip. Possible values: [0, 1].
            rpe_on: Where to apply relative positional encoding. Possible values: ['q', 'k', 'v', 'qk', 'kv', 'qkv'].
        """

        self.rpe_skip = rpe_skip

        if out_dim is None:
            out_dim = in_dim

        layers = torch.nn.ModuleList(
            [
                iRPETransformerLayer(
                    in_dim=in_dim if i == 0 else att_dim,
                    out_dim=out_dim if i == n_layers - 1 else att_dim,
                    att_dim=att_dim,
                    n_heads=n_heads,
                    use_mlp=use_mlp,
                    dropout=dropout,
                    rpe_ratio=rpe_ratio,
                    rpe_method=rpe_method,
                    rpe_mode=rpe_mode,
                    rpe_shared_head=rpe_shared_head,
                    rpe_skip=rpe_skip,
                    rpe_on=rpe_on,
                )
                for i in range(n_layers)
            ]
        )

        super(iRPETransformerEncoder, self).__init__(layers, add_self=add_self)

        self.norm = torch.nn.LayerNorm(out_dim)

    def forward(
        self,
        X: torch.Tensor,
        return_att: bool = False,
    ) -> torch.Tensor:
        """
        Forward method.

        Arguments:
            X: Input tensor of shape `(batch_size, bag_size, in_dim)`.
            return_att: If True, returns attention weights, of shape `(n_layers, batch_size, n_heads, bag_size, bag_size)`.

        Returns:
            Y: Output tensor of shape `(batch_size, bag_size, in_dim)`.
        """

        out = super().forward(X, return_att=return_att)
        if return_att:
            Y = out[0]  # (batch_size, bag_size, out_dim)
            att = out[1]  # (n_layers, batch_size, n_heads, bag_size, new_seq_len)
        else:
            Y = out  # (batch_size, bag_size, out_dim)
        Y = self.norm(Y)  # (batch_size, bag_size, att_dim)

        if return_att:
            return Y, att
        return Y
