import torch
from einops import rearrange


def moore_penrose_iter_pinv(x, iters=6):
    device = x.device

    abs_x = torch.abs(x)
    col = abs_x.sum(dim=-1)
    row = abs_x.sum(dim=-2)
    # each batch element is scaled by its own maxima, so that it does not depend on the rest of the batch
    batch_shape = x.shape[:1] if x.dim() > 2 else ()
    scale = col.reshape(*batch_shape, -1).amax(dim=-1) * row.reshape(
        *batch_shape, -1
    ).amax(dim=-1)
    scale = scale.reshape(*batch_shape, *([1] * (x.dim() - len(batch_shape))))
    z = rearrange(x, "... i j -> ... j i") / scale

    Id = torch.eye(x.shape[-1], device=device)
    Id = rearrange(Id, "i j -> () i j")

    for _ in range(iters):
        xz = x @ z
        z = 0.25 * z @ (13 * Id - (xz @ (15 * Id - (xz @ (7 * Id - xz)))))

    return z


class NystromAttention(torch.nn.Module):
    """
    Nystrom attention, as described in the paper [Nyströmformer: A Nyström-Based Algorithm for Approximating Self-Attention](https://arxiv.org/abs/2102.03902).

    Implementation based on the [official code](https://github.com/lucidrains/nystrom-attention).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = None,
        att_dim: int = 512,
        n_heads: int = 4,
        learn_weights: bool = True,
        n_landmarks: int = 256,
        pinv_iterations: int = 6,
    ):
        """
        Arguments:
            in_dim: Input dimension.
            out_dim: Output dimension. If None, out_dim = in_dim.
            att_dim: Attention dimension. Must be divisible by `n_heads`.
            n_heads: Number of heads.
            learn_weights: If True, learn the weights for query, key, and value. If False, q, k, and v are the same as the input, and therefore `in_dim` must be divisible by `n_heads`.
            n_landmarks: Number of landmarks.
            pinv_iterations: Number of iterations for Moore-Penrose pseudo-inverse.
        """
        super().__init__()

        if out_dim is None:
            out_dim = in_dim

        self.eps = 1e-8
        head_dim = att_dim // n_heads

        self.n_landmarks = n_landmarks
        self.pinv_iterations = pinv_iterations

        self.n_heads = n_heads
        self.scale_factor = head_dim**-0.5
        self.learn_weights = learn_weights
        if learn_weights:
            self.qkv_nn = torch.nn.Linear(in_dim, att_dim * 3, bias=False)
            self.head_dim = att_dim // n_heads
            assert att_dim % n_heads == 0, "att_dim must be divisible by n_heads"
        else:
            self.qkv_nn = None
            self.head_dim = in_dim // n_heads
            att_dim = in_dim
            assert in_dim % n_heads == 0, "in_dim must be divisible by n_heads"

        if out_dim != att_dim:
            self.out_proj = torch.nn.Linear(att_dim, out_dim)
        else:
            self.out_proj = torch.nn.Identity()

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
            mask: Mask tensor of shape `(batch_size, seq_len)`. Each sequence is processed as if its masked positions were not there.
            return_att: Whether to return attention weights.

        Returns:
            y: Output tensor of shape `(batch_size, seq_len, att_dim)`.
            att: Only returned when `return_att=True`. Attention weights of shape `(batch_size, n_heads, seq_len, seq_len)`.
        """

        batch_size, seq_len, _ = x.size()
        device = x.device

        # m = num_landmarks, h = heads, d = head_dim, iters = pinv_iterations, eps = eps

        if mask is None:
            mask = torch.ones(batch_size, seq_len, device=device)
        mask = mask.bool()

        # move the masked positions to the front, keeping the order of the rest

        has_masked = not bool(mask.all())
        if has_masked:
            order = torch.argsort(
                mask.int(), dim=1, stable=True
            )  # (batch_size, seq_len)
            x = torch.gather(x, 1, order[..., None].expand_as(x))
            mask = torch.gather(mask, 1, order)
            x = x * mask[..., None]

        # pad so that sequence can be evenly divided into m landmarks

        remainder = seq_len % self.n_landmarks
        if remainder > 0:
            padding = self.n_landmarks - remainder
            x = torch.nn.functional.pad(
                x, (0, 0, padding, 0), value=0
            )  # (batch_size, seq_len + padding, att_dim)
            mask = torch.nn.functional.pad(
                mask, (padding, 0), value=False
            )  # (batch_size, seq_len + padding)
        new_seq_len = x.size(1)

        # each sequence uses the positions it would use if it was alone in the batch:
        # its non-masked positions, preceded by the zeros needed to evenly divide them into m landmarks

        lm = (mask.sum(dim=1) + self.n_landmarks - 1) // self.n_landmarks
        lm = lm.clamp(min=1)  # (batch_size,)
        start = new_seq_len - lm * self.n_landmarks  # (batch_size,)
        position = torch.arange(new_seq_len, device=device)
        position = position[None, :] - start[:, None]  # (batch_size, new_seq_len)
        used = position >= 0  # (batch_size, new_seq_len)

        q, k, v = self._qkv(
            x
        )  # (batch_size, new_seq_len, att_dim), (batch_size, new_seq_len, att_dim), (batch_size, new_seq_len, att_dim)
        q = q.view(batch_size, new_seq_len, self.n_heads, -1).permute(
            0, 2, 1, 3
        )  # (batch_size, n_heads, new_seq_len, head_dim)
        k = k.view(batch_size, new_seq_len, self.n_heads, -1).permute(
            0, 2, 1, 3
        )  # (batch_size, n_heads, new_seq_len, head_dim)
        v = v.view(batch_size, new_seq_len, self.n_heads, -1).permute(
            0, 2, 1, 3
        )  # (batch_size, n_heads, new_seq_len, head_dim)

        q = q * self.scale_factor

        # generate landmarks by sum reduction of the lm consecutive positions assigned to each of them,
        # and then calculate mean. Positions that are not used are zero, so they add nothing

        landmark = (position // lm[:, None]).clamp(min=0)  # (batch_size, new_seq_len)
        landmark = landmark[:, None, :, None].expand_as(q)
        landmarks_shape = (batch_size, self.n_heads, self.n_landmarks, q.size(-1))
        divisor = lm[:, None, None, None]
        q_landmarks = (
            q.new_zeros(landmarks_shape).scatter_add(2, landmark, q) / divisor
        )  # (batch_size, n_heads, n_landmarks, head_dim)
        k_landmarks = (
            k.new_zeros(landmarks_shape).scatter_add(2, landmark, k) / divisor
        )  # (batch_size, n_heads, n_landmarks, head_dim)

        # similarities

        einops_eq = "... i d, ... j d -> ... i j"
        sim1 = torch.einsum(einops_eq, q, k_landmarks)
        sim2 = torch.einsum(einops_eq, q_landmarks, k_landmarks)
        sim3 = torch.einsum(einops_eq, q_landmarks, k)

        # masking: the positions a sequence does not use receive no attention

        if has_masked:
            mask_value = -torch.finfo(q.dtype).max
            sim3 = sim3.masked_fill(~used[:, None, None, :], mask_value)

        # eq (15) in the paper and aggregate values

        attn1 = sim1.softmax(dim=-1)
        attn2 = sim2.softmax(dim=-1)
        attn3 = sim3.softmax(dim=-1)

        attn2_inv = moore_penrose_iter_pinv(attn2, self.pinv_iterations)

        out = (attn1 @ attn2_inv) @ (
            attn3 @ v
        )  # (batch_size, n_heads, new_seq_len, head_dim)

        # add depth-wise conv residual of values

        # merge and combine heads

        out = rearrange(
            out, "b h n d -> b n (h d)", h=self.n_heads
        )  # (batch_size, new_seq_len, att_dim)
        out = self.out_proj(out)  # (batch_size, new_seq_len, att_dim)
        out = out[:, -seq_len:]  # remove padding

        # restore the original order, with zeros in the masked positions

        if has_masked:
            mask = mask[:, -seq_len:]  # (batch_size, seq_len)
            out = out * mask[..., None]
            out = torch.zeros_like(out).scatter(1, order[..., None].expand_as(out), out)

        if return_att:
            attn = attn1 @ attn2_inv @ attn3
            # remove padding
            attn = attn[:, :, -seq_len:, -seq_len:]
            if has_masked:
                attn = attn * mask[:, None, :, None] * mask[:, None, None, :]
                row_order = order[:, None, :, None].expand_as(attn)
                col_order = order[:, None, None, :].expand_as(attn)
                attn = torch.zeros_like(attn).scatter(2, row_order, attn)
                attn = torch.zeros_like(attn).scatter(3, col_order, attn)
            return out, attn
        else:
            return out
