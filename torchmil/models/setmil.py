import torch

from torchmil.models.mil_model import MILModel

from torchmil.nn.utils import get_feat_dim, SinusoidalPositionalEncodingND

from torchmil.nn.transformers import iRPETransformerEncoder, TransformerLayer

from torchmil.data import (
    seq_to_spatial,
    spatial_to_seq,
)


class T2TLayer(torch.nn.Module):
    r"""
    Tokens-to-Token (T2T) Transformer layer from [Tokens-to-Token ViT: Training Vision Transformers from Scratch on ImageNet](https://arxiv.org/abs/2101.11986)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = None,
        att_dim: int = 512,
        kernel_size: tuple[int, int] = (3, 3),
        stride: tuple[int, int] = (1, 1),
        padding: tuple[int, int] = (2, 2),
        dilation: tuple[int, int] = (1, 1),
        n_heads: int = 4,
        use_mlp: bool = True,
        dropout: float = 0.0,
    ):
        """
        Arguments:
            in_dim: Input dimension.
            out_dim: Output dimension. If None, output dimension will be `kernel_size[0] * kernel_size[1] * att_dim`.
            att_dim: Attention dimension.
            kernel_size: Kernel size.
            stride: Stride.
            padding: Padding.
            dilation: Dilation.
            n_heads: Number of heads.
            use_mlp: Whether to use feedforward layer.
            dropout: Dropout rate.
        """
        super().__init__()

        self.unfold = torch.nn.Unfold(
            kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation
        )

        self.transf_layer = TransformerLayer(
            in_dim=in_dim * kernel_size[0] * kernel_size[1],
            att_dim=att_dim,
            out_dim=att_dim,
            n_heads=n_heads,
            use_mlp=use_mlp,
            dropout=dropout,
        )

        if out_dim != att_dim:
            self.out_proj = torch.nn.Linear(att_dim, out_dim)
        else:
            self.out_proj = torch.nn.Identity()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            X: Input tensor of shape `(batch_size, in_dim, h, w)`.
        Returns:
            Y: Output tensor of shape `(batch_size, new_seq_len, out_dim)`. If `out_dim` is None, `out_dim` will be `att_dim * kernel_size[0] * kernel_size[1]`.
        """

        # unfold
        X = self.unfold(
            X
        )  # (batch_size, in_dim * kernel_size[0] * kernel_size[1], new_seq_len)
        X = X.transpose(
            1, 2
        )  # (batch_size, new_seq_len, in_dim * kernel_size[0] * kernel_size[1])

        # transformer layer
        X = self.transf_layer(X)  # (batch_size, new_seq_len, att_dim)

        # output projection
        X = self.out_proj(X)  # (batch_size, new_seq_len, out_dim)

        return X


class PMF(torch.nn.Module):
    r"""
    Pyramid Multi-Scale Fusion (PMF) module, proposed in the paper [SETMIL: Spatial Encoding Transformer-Based Multiple Instance Learning for Pathological Image Analysis](https://link.springer.com/chapter/10.1007/978-3-031-16434-7_7).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = None,
        att_dim: int = 512,
        kernel_list: list[tuple[int, int]] = [(3, 3), (5, 5), (7, 7)],
        stride_list: list[tuple[int, int]] = [(1, 1), (1, 1), (1, 1)],
        padding_list: list[tuple[int, int]] = [(1, 1), (2, 2), (3, 3)],
        dilation_list: list[tuple[int, int]] = [(1, 1), (1, 1), (1, 1)],
        n_heads: int = 4,
        use_mlp: bool = True,
        dropout: float = 0.0,
    ):
        """
        Arguments:
            in_dim: Input dimension.
            out_dim: Output dimension.
            att_dim: Attention dimension.
            kernel_list: List of kernel sizes.
            stride_list: List of strides.
            padding_list: List of paddings.
            dilation_list: List of dilations.
            n_heads: Number of heads.
            use_mlp: Whether to use feedforward layer.
            dropout: Dropout rate.
        """

        super().__init__()

        self.layers = torch.nn.ModuleList(
            [
                T2TLayer(
                    in_dim=in_dim,
                    att_dim=att_dim,
                    out_dim=att_dim,
                    kernel_size=kernel_list[i],
                    stride=stride_list[i],
                    padding=padding_list[i],
                    dilation=dilation_list[i],
                    n_heads=n_heads,
                    use_mlp=use_mlp,
                    dropout=dropout,
                )
                for i in range(len(kernel_list))
            ]
        )

        if out_dim is not None:
            self.out_proj = torch.nn.Linear(att_dim * len(kernel_list), out_dim)
        else:
            self.out_proj = torch.nn.Identity()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            X: Input tensor of shape `(batch_size, in_dim, h, w)`.
        Returns:
            Y: Output tensor of shape `(batch_size, new_seq_len, out_dim)`.
        """
        X_ = []
        for layer in self.layers:
            U = layer(X)  # (batch_size, L, att_dim)
            X_.append(U)
        X_ = torch.cat(
            X_, dim=2
        )  # (batch_size, new_seq_len, att_dim * len(kernel_list))
        X_ = self.out_proj(X_)  # (batch_size, new_seq_len, out_dim)
        return X_


class SETMIL(MILModel):
    r"""
    SETMIL: Spatial Encoding Transformer-Based Multiple Instance Learning for Pathological Image Analysis (SETMIL) model, proposed in the paper [SETMIL: Spatial Encoding Transformer-Based Multiple Instance Learning for Pathological Image Analysis](https://link.springer.com/chapter/10.1007/978-3-031-16434-7_7).

    Given an input bag $\mathbf{X} = \left[ \mathbf{x}_1, \ldots, \mathbf{x}_N \right]^\top \in \mathbb{R}^{N \times P}$, the model optionally applies a feature extractor, $\text{FeatExt}(\cdot)$, to transform the instance features: $\mathbf{X} = \text{FeatExt}(\mathbf{X}) \in \mathbb{R}^{N \times D}$.

    Then, the Pyramid Multi-Scale Fusion (PMF) module enriches the representation with multi-scale context information.
    The PMF module consists of three T2T modules with different kernel sizes, $k = 3, 5, 7$, concatenated along the feature dimension,

    $$\operatorname{PMF}\left( \mathbf{X} \right) = \text{Concat}(\text{T2T}_{k=3}(\mathbf{X}), \text{T2T}_{k=5}(\mathbf{X}), \text{T2T}_{k=7}(\mathbf{X})).$$

    See [T2T](https://arxiv.org/abs/2101.11986) and [T2TLayer](../nn/transformers/t2t.md) for further information.

    Then, the model applies a Spatial Encoding Transformer (SET), which consists of a stack of transformer layers with image Relative Positional Encoding (iRPE).
    See [iRPETransformer](../nn/transformers/irpe_transformer.md) for further information.

    Finally, using the class token computed by the SET module, the model predicts the bag label $\hat{Y}$ using a linear layer.

    **Note.** When `use_pmf=True`, the input bag is reshaped to a square shape, and the PMF module is applied.
    This modifies the bag structure unreversibly, and thus attention values cannot be computed.
    If `return_att=True`, the attention values will be set to zeros.
    """

    def __init__(
        self,
        in_shape: tuple,
        att_dim: int = 512,
        use_pmf: bool = False,
        pmf_n_heads: int = 4,
        pmf_use_mlp: bool = True,
        pmf_dropout: float = 0.0,
        pmf_kernel_list: list[tuple[int, int]] = [(3, 3), (5, 5), (7, 7)],
        pmf_stride_list: list[tuple[int, int]] = [(1, 1), (1, 1), (1, 1)],
        pmf_padding_list: list[tuple[int, int]] = [(1, 1), (2, 2), (3, 3)],
        pmf_dilation_list: list[tuple[int, int]] = [(1, 1), (1, 1), (1, 1)],
        set_n_layers: int = 1,
        set_n_heads: int = 4,
        set_use_mlp: bool = True,
        set_dropout: float = 0.0,
        rpe_ratio: float = 1.9,
        rpe_method: str = "product",
        rpe_mode: str = "contextual",
        rpe_shared_head: bool = True,
        rpe_skip: int = 1,
        rpe_on: str = "k",
        feat_ext: torch.nn.Module = torch.nn.Identity(),
        n_outputs: int = 1,
        criterion: torch.nn.Module = torch.nn.BCEWithLogitsLoss(),
    ) -> None:
        """
        Arguments:
            in_shape: Shape of input data expected by the feature extractor (excluding batch dimension).
            att_dim: Attention dimension used by the PMF and SET modules.
            use_pmf: If True, use Pyramid Multihead Feature (PMF) before the SET module.
            pmf_n_heads: Number of heads in the PMF module.
            pmf_use_mlp: If True, use MLP in the PMF module.
            pmf_dropout: Dropout rate in the PMF module.
            pmf_kernel_list: List of kernel sizes in the PMF module.
            pmf_stride_list: List of stride sizes in the PMF module.
            pmf_padding_list: List of padding sizes in the PMF module.
            pmf_dilation_list: List of dilation sizes in the PMF module.
            set_n_layers: Number of layers in the SET module.
            set_n_heads: Number of heads in the SET module.
            set_use_mlp: If True, use MLP in the SET module.
            set_dropout: Dropout rate in the SET module.
            rpe_ratio: Ratio for relative positional encoding.
            rpe_method: Method for relative positional encoding. Possible values: ['euc', 'quant', 'cross', 'product']
            rpe_mode: Mode for relative positional encoding. Possible values: [None, 'bias', 'contextual']
            rpe_shared_head: If True, share weights across different heads.
            rpe_skip: Number of tokens to skip in the relative positional encoding. Possible values: [0, 1].
            rpe_on: Where to apply relative positional encoding. Possible values: ['q', 'k', 'v', 'qk', 'kv', 'qkv'].
            feat_ext: Feature extractor.
            n_outputs: Number of outputs. By default, 1 (binary classification).
            criterion: Loss function. By default, Binary Cross-Entropy loss from logits.
        """
        super().__init__()
        self.num_outputs = n_outputs
        self.criterion = criterion

        self.feat_ext = feat_ext
        feat_dim = get_feat_dim(feat_ext, in_shape)

        if feat_dim != att_dim:
            self.proj = torch.nn.Linear(feat_dim, att_dim)
        else:
            self.proj = torch.nn.Identity()
        self.att_dim = att_dim

        self.pos_embed = SinusoidalPositionalEncodingND(1, att_dim)

        self.use_pmf = use_pmf
        if use_pmf:
            self.pmf = PMF(
                in_dim=att_dim,
                att_dim=att_dim,
                out_dim=att_dim,
                kernel_list=pmf_kernel_list,
                stride_list=pmf_stride_list,
                padding_list=pmf_padding_list,
                dilation_list=pmf_dilation_list,
                n_heads=pmf_n_heads,
                use_mlp=pmf_use_mlp,
                dropout=pmf_dropout,
            )

        self.se_transf = iRPETransformerEncoder(
            in_dim=att_dim,
            att_dim=att_dim,
            n_layers=set_n_layers,
            n_heads=set_n_heads,
            use_mlp=set_use_mlp,
            dropout=set_dropout,
            rpe_ratio=rpe_ratio,
            rpe_method=rpe_method,
            rpe_mode=rpe_mode,
            rpe_shared_head=rpe_shared_head,
            rpe_skip=rpe_skip,
            rpe_on=rpe_on,
        )

        self.cls_token = torch.nn.Parameter(torch.randn(1, 1, att_dim))

        self.classifier = torch.nn.Linear(in_features=att_dim, out_features=n_outputs)

    def _pad_to_square(self, X: torch.Tensor) -> torch.Tensor:
        """
        Pad tensor to square shape.

        Arguments:
            X: Input tensor of shape `(batch_size, bag_size, coord1, coord2)`.

        Returns:
            X: Padded tensor of shape `(batch_size, bag_size, max_dim, max_dim)`.
        """
        max_dim = max(X.size(-2), X.size(-1))
        pad_h = max_dim - X.size(-2)
        pad_w = max_dim - X.size(-1)
        X = torch.nn.functional.pad(X, (0, pad_w, 0, pad_h))
        return X

    def forward(
        self, X: torch.Tensor, coords: torch.Tensor, return_att: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Arguments:
            X: Bag features of shape `(batch_size, bag_size, feat_dim)`.
            coords: Coordinates of shape `(batch_size, bag_size, coord_dim)`.
            return_att: If True, returns attention values (before normalization) in addition to `Y_pred`.

        Returns:
            Y_pred: Bag label logits of shape `(batch_size,)`.
            att: Only returned when `return_att=True`. Attention values (before normalization) of shape (batch_size, bag_size).
        """

        assert coords.shape[2] == 2, "SETMIL only supports 2D coordinates."
        coords = coords.long()

        batch_size = X.size(0)
        orig_bag_size = X.size(1)

        X = self.feat_ext(X)  # (batch_size, bag_size, feat_dim)
        X = self.proj(X)  # (batch_size, bag_size, att_dim)

        X = seq_to_spatial(X, coords)  # (batch_size, coord1, coord2, att_dim)

        X = X.transpose(1, -1)  # (batch_size, att_dim, coord2, coord1)
        X = X.transpose(-1, -2)  # (batch_size, att_dim, coord1, coord2)
        X = self._pad_to_square(X)  # (batch_size, att_dim, max_coord, max_coord)
        max_coord = X.size(-1)

        if self.use_pmf:
            X = self.pmf(X)  # (batch_size, new_seq_len, att_dim)
        else:
            X = X.reshape(
                batch_size, -1, self.att_dim
            )  # (batch_size, bag_size, att_dim)

        # Add class token
        cls_token = self.cls_token.expand(
            batch_size, -1, -1
        )  # (batch_size, 1, att_dim)
        X = torch.cat([cls_token, X], dim=1)  # (batch_size, new_seq_len + 1, att_dim)

        pos = self.pos_embed(X)  # (batch_size, new_seq_len + 1, att_dim)
        X = X + pos  # (batch_size, new_seq_len + 1, att_dim)

        if return_att:
            if self.use_pmf:
                X = self.se_transf(X)  # (batch_size, new_seq_len + 1, att_dim)
                att = torch.zeros(batch_size, orig_bag_size, device=X.device)
            else:
                X, att = self.se_transf(
                    X, return_att=True
                )  # (batch_size, seq_len + 1, att_dim), (n_layers, batch_size, n_heads, seq_len+1, seq_len+1)

                att = att.mean(2)  # (n_layers, batch_size, seq_len + 1, seq_len + 1)
                att = att[-1, :, 0, 1:]  # (batch_size, seq_len)
                att = att.view(
                    batch_size, max_coord, max_coord
                )  # (batch_size, max_coord, max_coord)
                att = att.unsqueeze(-1)  # (batch_size, max_coord, max_coord, 1)
                att = spatial_to_seq(att, coords)  # (batch_size, bag_size, 1)
                att = att.squeeze(-1)  # (batch_size, bag_size)
        else:
            X = self.se_transf(X)  # (batch_size, bag_size' + 1, att_dim)

        z = X[:, 0]  # (batch_size, att_dim)

        Y_pred = self.classifier(z).squeeze(-1)  # (batch_size, 1)

        if return_att:
            return Y_pred, att
        else:
            return Y_pred

    def compute_loss(
        self,
        Y: torch.Tensor,
        X: torch.Tensor,
        coords: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute loss given true bag labels.

        Arguments:
            Y: Bag labels of shape `(batch_size,)`.
            X: Bag features of shape `(batch_size, bag_size, ...)`.
            coords: Coordinates of shape `(batch_size, bag_size, coord_dim)`.

        Returns:
            Y_pred: Bag label logits of shape `(batch_size,)`.
            loss_dict: Dictionary containing the loss value.
        """
        Y_pred = self.forward(X, coords, return_att=False)

        crit_loss = self.criterion(Y_pred.float(), Y.float())
        crit_name = self.criterion.__class__.__name__

        return Y_pred, {crit_name: crit_loss}

    def predict(
        self, X: torch.Tensor, coords: torch.Tensor, return_inst_pred: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict bag and (optionally) instance labels.

        Arguments:
            X: Bag features of shape `(batch_size, bag_size, ...)`.
            coords: Coordinates of shape `(batch_size, bag_size, coord_dim)`.
            return_inst_pred: If `True`, returns instance labels predictions, in addition to bag label predictions.

        Returns:
            Y_pred: Bag label logits of shape `(batch_size,)`.
            y_inst_pred: If `return_inst_pred=True`, returns instance labels predictions of shape `(batch_size, bag_size)`.
        """
        return self.forward(X, coords, return_att=return_inst_pred)
