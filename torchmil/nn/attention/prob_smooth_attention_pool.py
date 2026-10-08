import torch
from torch import Tensor

from torchmil.nn.utils import masked_softmax, LazyLinear


class ProbSmoothAttentionPool(torch.nn.Module):
    r"""
    Probabilistic Smooth Attention Pooling, proposed in in [Probabilistic Smooth Attention for Deep Multiple Instance Learning in Medical Imaging]() and [Smooth Attention for Deep Multiple Instance Learning: Application to CT Intracranial Hemorrhage Detection](https://arxiv.org/abs/2307.09457)

    Given an input bag $\mathbf{X} = \left[ \mathbf{x}_1, \ldots, \mathbf{x}_N \right]^\top \in \mathbb{R}^{N \times \texttt{in_dim}}$,
    this model computes an attention distribution $q(\mathbf{f} \mid \mathbf{X}) = \mathcal{N}\left(\mathbf{f} \mid \mathbf{\mu}_{\mathbf{f}}, \operatorname{diag}(\mathbf{\sigma}_{\mathbf{f}}^2) \right)$, where:

    \begin{gather}
        \mathbf{H} = \operatorname{MLP}(\mathbf{X}) \in \mathbb{R}^{N \times 2\texttt{att_dim}}, \\
        \mathbf{\mu}_{\mathbf{f}} = \mathbf{H}\mathbf{w}_{\mu} \in \mathbb{R}^{N}, \\
        \log \mathbf{\sigma}_{\mathbf{f}}^2 = \mathbf{H}\mathbf{w}_{\sigma} \in \mathbb{R}^{N},
    \end{gather}

    where $\operatorname{MLP}$ is a multi-layer perceptron, and $\mathbf{w}_{\mu},\mathbf{w}_{\sigma} \in \mathbb{R}^{2\texttt{att_dim} \times 1}$.
    If `covar_mode='zero'`, the variance vector $\mathbf{\sigma}_{\mathbf{f}}^2$ is set to zero, resulting in a deterministic attention distribution.

    Then, $M$ samples from the attention distribution are drawn as $\widehat{\mathbf{f}}^{(m)} \sim q(\mathbf{f} \mid \mathbf{X})$.
    With these samples, the bag representation is computed as:

    $$
    \widehat{\mathbf{z}} = \operatorname{Softmax}(\widehat{\mathbf{F}}) \mathbf{X} \in \mathbb{R}^{\texttt{in_dim} \times M},
    $$

    where $\widehat{\mathbf{F}} = \left[ \widehat{\mathbf{f}}^{(1)}, \ldots, \widehat{\mathbf{f}}^{(M)} \right]^\top \in \mathbb{R}^{N \times M}$.

    **Kullback-Leibler Divergence.** Given a bag with adjancency matrix $\mathbf{A}$, the KL divergence between the attention distribution and the prior distribution is computed as:

    $$
        \ell_{\text{KL}} =
            \begin{cases}
                \mathbf{\mu}_{\mathbf{f}}^\top \mathbf{L} \mathbf{\mu}_{\mathbf{f}} \quad & \text{if } \texttt{covar_mode='zero'}, \\
                \mathbf{\mu}_{\mathbf{f}}^\top \mathbf{L} \mathbf{\mu}_{\mathbf{f}} + \operatorname{Tr}(\mathbf{L} \mathbf{\Sigma}_{\mathbf{f}}) - \frac{1}{2}\log \det( \mathbf{\Sigma}_{\mathbf{f}} ) + \operatorname{const} \quad & \text{if } \texttt{covar_mode='diag'}, \\
            \end{cases}
    $$

    where $\operatorname{const}$ is a constant term that does not depend on the parameters, $\mathbf{\Sigma}_{\mathbf{f}} = \operatorname{diag}(\mathbf{\sigma}_{\mathbf{f}}^2)$, $\mathbf{L} = \mathbf{D} - \mathbf{A}$ is the graph Laplacian matrix, and $\mathbf{D}$ is the degree matrix of $\mathbf{A}$.
    """

    def __init__(
        self,
        in_dim: int = None,
        att_dim: int = 128,
        covar_mode: str = "diag",
        n_samples_train: int = 1000,
        n_samples_test: int = 5000,
    ) -> None:
        """
        Arguments:
            in_dim: Input dimension. If not provided, it will be lazily initialized.
            att_dim: Attention dimension.
            covar_mode: Covariance mode. Must be 'diag' or 'zero'.
            n_samples_train: Number of samples during training.
            n_samples_test: Number of samples during testing.
        """
        super(ProbSmoothAttentionPool, self).__init__()
        self.covar_mode = covar_mode
        self.n_samples_train = n_samples_train
        self.n_samples_test = n_samples_test

        if self.covar_mode not in ["diag", "zero"]:
            raise ValueError("covar_mode must be 'diag' or 'zero'")

        self.in_mlp = torch.nn.Sequential(
            LazyLinear(in_dim, 2 * att_dim),
            torch.nn.GELU(),
            torch.nn.Linear(2 * att_dim, 2 * att_dim),
            torch.nn.GELU(),
        )

        self.mu_f_nn = torch.nn.Linear(2 * att_dim, 1)

        if self.covar_mode == "diag":
            self.log_diag_Sigma_nn = torch.nn.Linear(2 * att_dim, 1)

        self.eps = 1e-6

    def _sample_f(
        self, mu_f: Tensor, log_diag_Sigma_f: Tensor, n_samples: int = 1
    ) -> Tensor:
        """
        Arguments:
            mu_f: Mean of q(f) of shape `(batch_size, bag_size, 1)`.
            log_diag_Sigma_f: Log diagonal of covariance of q(f) of shape `(batch_size, bag_size, 1)`.
            n_samples: Number of samples to draw.

        Returns:
            f: Sampled f of shape `(batch_size, bag_size, n_samples)`.
        """
        batch_size = mu_f.shape[0]

        if self.covar_mode == "diag":
            bag_size = mu_f.shape[1]
            random_sample = torch.randn(
                batch_size, bag_size, n_samples, device=mu_f.device
            )  # (batch_size, bag_size, n_samples)
            sqrt_diag_Sigma_f = torch.exp(
                0.5 * log_diag_Sigma_f
            )  # (batch_size, bag_size, 1)
            f = (
                mu_f + sqrt_diag_Sigma_f * random_sample
            )  # (batch_size, bag_size, n_samples)
            f = torch.clip(f, -20, 20)
        else:
            f = mu_f
        return f

    def _kl_div(
        self, mu_f: Tensor, log_diag_Sigma_f: Tensor, adj_mat: Tensor, mask: Tensor
    ) -> Tensor:
        """
        Arguments:
            mu_f: Mean of the attention distribution of shape `(batch_size, bag_size, 1)`.
            log_diag_Sigma_f: Log diagonal of covariance of the attention distribution of shape `(batch_size, bag_size, 1)`.
            adj_mat: Adjacency matrix of shape `(batch_size, bag_size, bag_size)`.
            mask: Mask of shape `(batch_size, bag_size, 1)`.

        Returns:
            kl_div: KL divergence of shape `()`.
        """

        mu_f = mu_f.masked_fill(mask == 0, 0)
        bag_size = mask.sum(dim=(1, 2))
        inv_bag_size = 1.0 / bag_size

        if not adj_mat.is_sparse:
            adj_mat_dense = adj_mat
            diag_adj = torch.diagonal(adj_mat, dim1=1, dim2=2).unsqueeze(
                dim=-1
            )  # (batch_size, bag_size, 1)
        else:
            adj_mat_dense = adj_mat.to("cpu").coalesce().to_dense()
            diag_adj = (
                torch.diagonal(adj_mat_dense, dim1=1, dim2=2)
                .unsqueeze(dim=-1)
                .to(mu_f.device)
            )  # (batch_size, bag_size, 1)

        muT_mu = torch.sum(mu_f**2, dim=(1, 2))  # (batch_size,)
        adj_mat_mu = torch.bmm(adj_mat, mu_f)  # (batch_size, bag_size, 1)
        muT_adjmat_mu = torch.bmm(mu_f.transpose(1, 2), adj_mat_mu).squeeze(
            1, 2
        )  # (batch_size,)

        muT_lap_mu = inv_bag_size * (muT_mu - muT_adjmat_mu)  # (batch_size,)

        if self.covar_mode == "full":
            raise NotImplementedError("covar_mode='full' is not implemented yet")
        elif self.covar_mode == "diag":
            log_diag_Sigma_f = log_diag_Sigma_f.masked_fill(mask == 0, 0)
            diag_Sigma = torch.exp(log_diag_Sigma_f) * mask  # (batch_size, bag_size, 1)
            tr_Sigma = inv_bag_size * torch.sum(diag_Sigma, dim=(1, 2))  # (batch_size,)
            tr_adj_Sigma = inv_bag_size * torch.sum(
                diag_adj * diag_Sigma, dim=(1, 2)
            )  # (batch_size,)
            log_det_Sigma = inv_bag_size * torch.sum(
                log_diag_Sigma_f, dim=(1, 2)
            )  # (batch_size,)
        else:
            tr_Sigma = torch.zeros((1,), device=mu_f.device)
            tr_adj_Sigma = torch.zeros((1,), device=mu_f.device)
            log_det_Sigma = torch.zeros((1,), device=mu_f.device)

        kl_div = torch.mean(
            muT_lap_mu + tr_Sigma - tr_adj_Sigma - 0.5 * log_det_Sigma
        )  # ()

        return kl_div

    def forward(
        self,
        X: Tensor,
        adj: Tensor = None,
        mask: Tensor = None,
        return_att_samples: bool = False,
        return_att_dist: bool = False,
        return_kl_div: bool = False,
        n_samples: int = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        In the following, if `covar_mode='zero'` then `n_samples` is automatically set to 1 and `diag_Sigma_f` is set to None.

        Arguments:
            X: Bag features of shape `(batch_size, bag_size, dim)`.
            mask: Mask of shape `(batch_size, bag_size)`.
            adj: Adjacency matrix of shape `(batch_size, bag_size, bag_size)`. Only required when `return_kl_div=True`.
            return_att_samples: If True, returns samples from the attention distribution `f` in addition to `z`.
            return_att_dist: If True, returns the attention distribution (`mu_f`, `diag_Sigma_f`) in addition to `z`.
            return_kl_div: If True, returns the KL divergence between the attention distribution and the prior distribution.
            n_samples: Number of samples to draw. If not provided, it will use `n_samples_train` during training and `n_samples_test` during testing.

        Returns:
            z: Bag representation of shape `(batch_size, dim, n_samples)`.
            f: Samples from the attention distribution of shape `(batch_size, bag_size, n_samples)`. Only returned when `return_att_samples=True`.
            mu_f: Mean of the attention distribution of shape `(batch_size, bag_size, 1)`. Only returned when `return_att_dist=True`.
            diag_Sigma_f: Covariance of the attention distribution of shape `(batch_size, bag_size, 1)`. Only returned when `return_att_dist=True`.
            kl_div: KL divergence between the attention distribution and the prior distribution, of shape `()`. Only returned when `return_kl_div=True`.
        """

        if n_samples is None:
            if self.training:
                n_samples = self.n_samples_train
            else:
                n_samples = self.n_samples_test

        batch_size = X.shape[0]
        bag_size = X.shape[1]

        if mask is None:
            mask = torch.ones(
                batch_size, bag_size, device=X.device
            )  # (batch_size, bag_size)
        mask = mask.unsqueeze(dim=-1)  # (batch_size, bag_size, 1)

        H = self.in_mlp(X)  # (batch_size, bag_size, 2*att_dim)
        mu_f = self.mu_f_nn(H)  # (batch_size, bag_size, 1)
        if self.covar_mode == "diag":
            log_diag_Sigma_f = self.log_diag_Sigma_nn(H)  # (batch_size, bag_size, 1)
        else:
            log_diag_Sigma_f = None
        # sample from q(f)
        f = self._sample_f(
            mu_f, log_diag_Sigma_f, n_samples
        )  # (batch_size, bag_size, n_samples)

        s = masked_softmax(f, mask)  # (batch_size, bag_size, n_samples)

        z = torch.bmm(X.transpose(1, 2), s)  # (batch_size, d, n_samples)

        if return_kl_div:
            kl_div = self._kl_div(mu_f, log_diag_Sigma_f, adj, mask)  # ()
            if return_att_samples:
                if return_att_dist:
                    return z, f, mu_f, log_diag_Sigma_f, kl_div
                else:
                    return z, f, kl_div
            elif return_att_dist:
                return z, mu_f, log_diag_Sigma_f, kl_div
            else:
                return z, kl_div
        else:
            if return_att_samples:
                if return_att_dist:
                    return z, f, mu_f, log_diag_Sigma_f
                else:
                    return z, f
            elif return_att_dist:
                return z, mu_f, log_diag_Sigma_f
            else:
                return z
