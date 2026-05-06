import torch
import numpy as np


class LazyLinear(torch.nn.Module):
    """
    Lazy Linear layer. Extends `torch.nn.Linear` with lazy initialization.
    """

    def __init__(
        self, in_features=None, out_features=512, bias=True, device=None, dtype=None
    ):
        super().__init__()

        if in_features is not None:
            self.module = torch.nn.Linear(
                in_features, out_features, bias=bias, device=device, dtype=dtype
            )
        else:
            self.module = torch.nn.LazyLinear(
                out_features, bias=bias, device=device, dtype=dtype
            )

    def forward(self, x):
        return self.module(x)


def masked_softmax(
    X: torch.Tensor,
    mask: torch.Tensor = None,
) -> torch.Tensor:
    """
    Compute masked softmax along the second dimension.

    Arguments:
        X (Tensor): Input tensor of shape `(batch_size, N, ...)`.
        mask (Tensor): Mask of shape `(batch_size, N)`. If None, no masking is applied.

    Returns:
        Tensor: Masked softmax of shape `(batch_size, N, ...)`.
    """

    if mask is None:
        return torch.nn.functional.softmax(X, dim=1)

    # Ensure mask is of the same shape as X
    if mask.dim() < X.dim():
        mask = mask.unsqueeze(-1)

    # exp_X = torch.exp(X)
    # exp_X_masked = exp_X * mask
    # sum_exp_X_masked = exp_X_masked.sum(dim=1, keepdim=True)
    # softmax_X = exp_X_masked / (sum_exp_X_masked + 1e-8)
    # return softmax_X

    X_masked = X.masked_fill(mask == 0, -float("inf"))

    return torch.nn.functional.softmax(X_masked, dim=1)


class MaskedSoftmax(torch.nn.Module):
    """
    Compute masked softmax along the second dimension.
    """

    def __init__(self):
        super().__init__()

    def forward(self, X, mask):
        """
        Forward method.

        Arguments:
            X: Input tensor of shape `(batch_size, N, ...)`.
            mask: Mask tensor of shape `(batch_size, N)`.

        Returns:
            Tensor: Masked softmax of shape `(batch_size, N, ...)`.
        """
        return masked_softmax(X, mask)


def get_feat_dim(feat_ext: torch.nn.Module, input_shape: tuple[int, ...]) -> int:
    """
    Get feature dimension of a feature extractor.

    Arguments:
        feat_ext (torch.nn.Module): Feature extractor.
        input_shape (tuple): Input shape of the feature extractor.
    """
    with torch.no_grad():
        return feat_ext(torch.zeros((1, *input_shape))).shape[-1]


class SinusoidalPositionalEncodingND(torch.nn.Module):
    def __init__(self, n_dim, channels, dtype_override=None):
        """
        Positional encoding for tensors of arbitrary dimensions.

        Arguments:
            n_dim (int): Number of dimensions.
            channels (int): Number of channels.
            dtype_override (torch.dtype): Data type override.
        """
        super(SinusoidalPositionalEncodingND, self).__init__()
        self.n_dim = n_dim
        self.org_channels = channels
        channels = int(np.ceil(channels / (2 * n_dim)) * 2)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.dtype_override = dtype_override
        self.channels = channels

    def _get_embedding(self, sin_inp):
        emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
        return torch.flatten(emb, -2, -1)

    def forward(self, tensor):
        """
        Arguments:
            tensor (Tensor): Input tensor of shape `(batch_size, l1, l2, ..., lN, channels)`.

        Returns:
            Tensor: Positional encoding of shape `(batch_size, l1, l2, ..., lN, channels)`.
        """
        if len(tensor.shape) != self.n_dim + 2:
            raise RuntimeError("The input tensor has to be {}d!".format(self.n_dim + 2))

        shape = tensor.shape

        orig_ch = shape[-1]
        emb_shape = list(shape)[1:]
        emb_shape[-1] = self.channels * self.n_dim

        emb = torch.zeros(
            emb_shape,
            device=tensor.device,
            dtype=(
                self.dtype_override if self.dtype_override is not None else tensor.dtype
            ),
        )

        for i in range(self.n_dim):
            pos = torch.arange(
                shape[i + 1], device=tensor.device, dtype=self.inv_freq.dtype
            )
            sin_inp = torch.einsum("i,j->ij", pos, self.inv_freq)
            emb_i = self._get_embedding(sin_inp)
            for _ in range(self.n_dim - i - 1):
                emb_i = emb_i.unsqueeze(1)
            emb[..., i * self.channels : (i + 1) * self.channels] = emb_i

        return emb[None, ..., :orig_ch].repeat(
            shape[0], *(1 for _ in range(self.n_dim)), 1
        )


def log_sum_exp(x):
    """
    Compute log(sum(exp(x), 1)) in a numerically stable way.

    Arguments:
        x: Input tensor of shape `(batch_size, n_classes)`.

    Returns:
        log_sum_exp: Log sum exp of shape `(batch_size,)
    """
    max_score, _ = x.max(1)
    return max_score + torch.log(torch.sum(torch.exp(x - max_score[:, None]), 1))


def delta(y, labels, alpha=None):
    """
    Compute zero-one loss matrix for a vector of ground truth y

    Arguments:
        y: Ground truth labels, of shape `(batch_size, n_classes)`.
        labels: Possible labels, of shape `(n_classes,)`.
        alpha: Regularization parameter.

    Returns:
        delta: Zero-one loss matrix of shape `(batch_size, n_classes)`.
    """

    if isinstance(y, torch.autograd.Variable):
        labels = torch.autograd.Variable(labels, requires_grad=False).to(y.device)

    delta = torch.ne(y[:, None], labels[None, :]).float()

    if alpha is not None:
        delta = alpha * delta
    return delta


def detect_large(x, k, tau, thresh):
    top, _ = x.topk(k + 1, 1)
    # switch to hard top-k if (k+1)-largest element is much smaller than k-largest element
    hard = torch.ge(top[:, k - 1] - top[:, k], k * tau * np.log(thresh)).detach()
    smooth = hard.eq(0)
    return smooth, hard


class SmoothTop1SVM(torch.nn.Module):
    """
    Smooth Top-1 SVM loss, as described in [Smooth Loss Functions for Deep Top-k Classification](https://arxiv.org/abs/1802.07595).
    Implementation adapted from [the original code](https://github.com/oval-group/smooth-topk).
    """

    def __init__(self, n_classes: int, alpha: float = 1.0, tau: float = 1.0) -> None:
        """
        Arguments:
            n_classes: Number of classes.
            alpha: Regularization parameter.
            tau: Temperature parameter.
        """
        # super(SmoothTop1SVM, self).__init__(n_classes=n_classes, alpha=alpha)
        super().__init__()
        self.alpha = alpha
        self.n_classes = n_classes
        self.tau = tau
        self.thresh = 1e3
        self.labels = torch.from_numpy(np.arange(n_classes))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Arguments:
            x: Input tensor of shape `(batch_size, n_classes)`. If `n_classes=1`, the tensor is assumed to be the positive class score.
            y: Target tensor of shape `(batch_size,)`.

        Returns:
            loss: Loss tensor of shape `(batch_size,)`.
        """

        if y.dim() > 1:
            y = y.argmax(dim=1)

        # if x.shape[1] == 1:
        #     x = torch.cat([x, -x], 1) # add dummy dimension for binary classification

        smooth, hard = detect_large(x, 1, self.tau, self.thresh)

        loss = 0
        if smooth.data.sum():
            x_s, y_s = x[smooth], y[smooth]
            x_s = x_s.view(-1, x.size(1))
            loss += self.smooth_loss(x_s, y_s).sum() / x.size(0)
        if hard.data.sum():
            x_h, y_h = x[hard], y[hard]
            x_h = x_h.view(-1, x.size(1))
            loss += self.hard_loss(x_h, y_h).sum() / x.size(0)

        return loss

    def hard_loss(self, x, y):
        """
        Compute hard loss.

        Arguments:
            x: Input tensor of shape `(batch_size, n_classes)`.
            y: Target tensor of shape `(batch_size,)`.

        Returns:
            loss: Hard loss tensor of shape `(batch_size,)`.
        """

        y = y.long().view(-1, 1)
        # max oracle
        max_, _ = (x + delta(y.squeeze(1), self.labels, self.alpha)).max(1)
        # subtract ground truth
        loss = max_ - x.gather(1, y).squeeze()
        return loss

    def smooth_loss(self, x, y):
        """
        Compute smooth loss.

        Arguments:
            x: Input tensor of shape `(batch_size, n_classes)`.
            y: Target tensor of shape `(batch_size,)`.

        Returns:
            loss: Smooth loss tensor of shape `(batch_size,)`.
        """

        y = y.long().view(-1, 1)
        # add loss term and subtract ground truth score
        x = x + delta(y.squeeze(1), self.labels, self.alpha) - x.gather(1, y)
        # compute loss
        loss = self.tau * log_sum_exp(x / self.tau)

        return loss
