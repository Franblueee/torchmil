import torch

from torch.nn.attention import SDPBackend

SDP_BACKEND = [
    SDPBackend.MATH,
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION,
    SDPBackend.CUDNN_ATTENTION,
]


class MultiheadSelfAttention(torch.nn.Module):
    r"""
    The Multihead Self Attention module, as described in [Attention is All You Need](https://arxiv.org/abs/1706.03762).

    Given an input bag $\mathbf{X} = \left[ \mathbf{x}_1, \ldots, \mathbf{x}_N \right]^\top \in \mathbb{R}^{N \times \texttt{in_dim}}$,
    this module computes:

    \begin{gather*}
    \mathbf{Q} = \mathbf{X}\mathbf{W}_Q, \quad \mathbf{K} = \mathbf{X}\mathbf{W}_K, \quad \mathbf{V} = \mathbf{X}\mathbf{W}_V,\\
    \mathbf{Y} = \operatorname{Softmax}\left( \frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d}} \right) \mathbf{V},
    \end{gather*}

    where $d = \texttt{att_dim}$ and $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V \in \mathbb{R}^{\texttt{in_dim} \times \texttt{att_dim}}$ are learnable weight matrices.

    If $\texttt{out_dim} \neq \texttt{att_dim}$, $\mathbf{Y}$ is passed through a linear layer with output dimension $\texttt{out_dim}$.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = None,
        att_dim: int = 512,
        n_heads: int = 4,
        dropout: float = 0.0,
        learn_weights: bool = True,
    ):
        """
        Arguments:
            in_dim: Input dimension.
            att_dim: Attention dimension, must be divisible by `n_heads`.
            out_dim: Output dimension. If None, `out_dim` = `in_dim`.
            n_heads: Number of heads.
            dropout: Dropout rate.
            learn_weights: If True, learn the weights for query, key, and value. If False, q, k, and v are the same as the input, and therefore `in_dim` must be divisible by `n_heads`.
        """
        super(MultiheadSelfAttention, self).__init__()
        if out_dim is None:
            out_dim = in_dim
        self.n_heads = n_heads
        self.dropout = dropout
        self.learn_weights = learn_weights
        if learn_weights:
            self.qkv_nn = torch.nn.Linear(in_dim, 3 * att_dim, bias=False)
            self.head_dim = att_dim // n_heads
            assert att_dim % n_heads == 0, "att_dim must be divisible by n_heads"
        else:
            self.qkv_nn = None
            self.head_dim = in_dim // n_heads
            att_dim = in_dim
            assert in_dim % n_heads == 0, "in_dim must be divisible by n_heads"
        self.att_dim = att_dim

        if out_dim != att_dim:
            self.out_proj = torch.nn.Linear(att_dim, out_dim)
        else:
            self.out_proj = torch.nn.Identity()

    def _scaled_dot_product_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None,
        return_att: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """

        Scaled dot product attention.

        Arguments:
            query: Query tensor of shape `(batch_size, n_heads, seq_len, head_dim)`.
            key: Key tensor of shape `(batch_size, n_heads, seq_len, head_dim)`.
            value: Value tensor of shape `(batch_size, n_heads, seq_len, head_dim)`.
            mask: Mask tensor of shape `(batch_size, seq_len)`.
            return_att: Whether to return the attention matrices.
        Returns:
            out: Output tensor of shape `(batch_size, n_heads, seq_len, head_dim)`.
        """

        if mask is not None:
            # the following is equivalent to mask.unsqueeze(-1) @ mask.unsqueeze(-1).transpose(1, 2)
            mask = (
                mask[:, None, :, None] * mask[:, None, None, :]
            )  # (batch_size, 1, seq_len, seq_len)
            mask = mask.bool()  # (batch_size, 1, seq_len, seq_len)

        if return_att:
            query = query / (self.head_dim**0.5)
            qk = torch.einsum(
                "bhqd,bhkd->bhqk", query, key
            )  # (batch_size, n_heads, seq_len, seq_len)
            if mask is not None:
                qk.masked_fill_(mask, float("-inf"))
            att = torch.nn.functional.softmax(qk, dim=-1)
            att_d = torch.nn.functional.dropout(
                att, p=self.dropout, training=self.training
            )
            out = torch.einsum(
                "bhqk,bhkd->bhqd", att_d, value
            )  # (batch_size, n_heads, seq_len, head_dim)
            return out, att
        else:
            with torch.nn.attention.sdpa_kernel(SDP_BACKEND):
                out = torch.nn.functional.scaled_dot_product_attention(
                    query, key, value, mask, self.dropout if self.training else 0.0
                )  # (batch_size, n_heads, seq_len, head_dim)
            return out

    def _qkv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute query, key, and value tensors.

        Arguments:
            x: Input tensor of shape `(batch_size, seq_len, in_dim)`.
        Returns:
            query: Query tensor of shape `(batch_size, seq_len, att_dim)`.
            key: Key tensor of shape `(batch_size, seq_len, att_dim)`.
            value: Value tensor of shape `(batch_size, seq_len, att_dim)`.
        """
        if self.learn_weights:
            q, k, v = self.qkv_nn(x).chunk(
                3, dim=-1
            )  # (batch_size, seq_len, att_dim), (batch_size, seq_len, att_dim), (batch_size, seq_len, att_dim)
        else:
            q = k = v = x
        return q, k, v

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor = None, return_att: bool = False
    ) -> torch.Tensor:
        """
        Forward pass.

        Arguments:
            x: Input tensor of shape `(batch_size, seq_len, in_dim)`.
            mask: Mask tensor of shape `(batch_size, seq_len)`.
        Returns:
            y: Output tensor of shape `(batch_size, seq_len, att_dim)`.
            att: Only returned when `return_att=True`. Attention weights of shape `(batch_size, n_heads, seq_len, seq_len)`.
        """
        batch_size, seq_len, _ = x.size()
        query, key, value = self._qkv(
            x
        )  # (batch_size, seq_len, att_dim), (batch_size, seq_len, att_dim), (batch_size, seq_len, att_dim)
        query = query.view(batch_size, seq_len, self.n_heads, self.head_dim).permute(
            0, 2, 1, 3
        )  # (batch_size, n_heads, seq_len, head_dim)
        key = key.view(batch_size, seq_len, self.n_heads, self.head_dim).permute(
            0, 2, 1, 3
        )  # (batch_size, n_heads, seq_len, head_dim)
        value = value.view(batch_size, seq_len, self.n_heads, self.head_dim).permute(
            0, 2, 1, 3
        )  # (batch_size, n_heads, seq_len, head_dim)
        if return_att:
            y, att = self._scaled_dot_product_attention(
                query, key, value, mask, return_att
            )
            y = (
                y.permute(0, 2, 1, 3)
                .contiguous()
                .view(batch_size, seq_len, self.att_dim)
            )  # (batch_size, seq_len, att_dim)
            y = self.out_proj(y)
            return y, att
        else:
            y = self._scaled_dot_product_attention(
                query, key, value, mask
            )  # (batch_size, n_heads, seq_len, head_dim)
            y = (
                y.permute(0, 2, 1, 3)
                .contiguous()
                .view(batch_size, seq_len, self.att_dim)
            )  # (batch_size, seq_len, att_dim)
            y = self.out_proj(y)
            return y
