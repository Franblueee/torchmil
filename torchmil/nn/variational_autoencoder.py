import torch
from .mlp import MLP
import numpy as np


class VariationalAutoEncoder(torch.nn.Module):
    r"""
    Variational Autoencoder (VAE) model for learning latent representations.

    The VAE learns a latent representation $\mathbf{z}$ of input data $\mathbf{x}$ by maximizing the Evidence Lower Bound (ELBO):

    $$
    \mathcal{L}(\theta, \phi; \mathbf{x}) = \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})} [\log p_\theta(\mathbf{x}|\mathbf{z})] - \text{KL}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))
    $$

    where $q_\phi(\mathbf{z}|\mathbf{x})$ is the encoder (posterior) and $p_\theta(\mathbf{x}|\mathbf{z})$ is the decoder (likelihood).
    Both the encoder and decoder are implemented as MLPs.
    """

    def __init__(
        self,
        input_shape: tuple[int] = (512,),
        layer_sizes: list[int] = [128, 64],
        activations: list[str] = ["relu", "None"],
        covar_mode: str = "single",
        jitter: float = 1e-7,
    ) -> None:
        """
        Arguments:
            input_shape: Shape of input data (excluding batch dimension).
            layer_sizes: List of hidden layer sizes for the encoder (decoder mirrors this).
            activations: List of activation functions for each layer. Must have same length as layer_sizes.
            covar_mode: Covariance mode for the variational distributions. Options: 'single', 'diagonal'.
            jitter: Small value added to log_std for numerical stability.
        """
        super().__init__()

        self.input_dim = input_shape
        self.output_size = input_shape[0]
        self.jitter = jitter
        self.covar_mode = covar_mode
        self.layer_sizes = layer_sizes

        dimensions_enc = [input_shape[0]] + layer_sizes
        dimensions_dec = layer_sizes[::-1] + [input_shape[0]]

        # Compute Variance dimensions
        if covar_mode == "single":
            self.d_var_enc, self.d_var_dec = 1, 1
        elif covar_mode == "diagonal":
            self.d_var_enc, self.d_var_dec = layer_sizes[-1], input_shape[0]
        else:
            raise NotImplementedError(
                f"{covar_mode} covar mode not valid. Current implementations: single/diagonal"
            )

        dimensions_enc[-1] += self.d_var_enc
        dimensions_dec[-1] += self.d_var_dec

        self.encoder = MLP(
            input_size=dimensions_enc[0],
            linear_sizes=dimensions_enc[1:],
            activations=["relu" for i in range(len(dimensions_enc) - 2)] + ["None"],
        )
        self.decoder = MLP(
            input_size=dimensions_dec[0],
            linear_sizes=dimensions_dec[1:],
            activations=["relu" for i in range(len(dimensions_dec) - 2)] + ["None"],
        )

    def get_reparameterized_samples(
        self, mean: torch.Tensor, log_std: torch.Tensor, n_samples: int = 1
    ) -> torch.Tensor:
        """
        Generate reparameterized samples using the reparameterization trick.

        Arguments:
            mean: Mean of the distribution of shape `(batch_size, latent_dim)`.
            log_std: Log standard deviation of shape `(batch_size, d_var_enc)`.
            n_samples: Number of samples to generate.

        Returns:
            Reparameterized samples of shape `(batch_size, n_samples, latent_dim)`.
        """

        # Obtain samples
        samples = torch.normal(
            0.0, 1.0, size=(mean.shape[0], mean.shape[-1], n_samples)
        ).to(mean.device)  # (batch_size, latent_dim, n_samples)

        # Reparameterized samples, add jitter
        log_std = log_std + self.jitter
        rep_samples = samples * torch.exp(log_std).unsqueeze(-1) + mean.unsqueeze(
            -1
        )  # (batch_size, latent_dim, n_samples)
        rep_samples = rep_samples.transpose(1, 2)  # (batch_size, n_samples, latent_dim)

        return rep_samples

    def get_raw_output_enc(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        Compute the mean and log standard deviation of the posterior distribution $q(\mathbf{z}\mid \mathbf{x})$.

        The posterior distribution is parameterized as:

        $q(\mathbf{z} \mid \mathbf{x}) = \mathcal N(\mathbf{x} \mid \mu(\mathbf{x}), \sigma(\mathbf{x}) * \mathbf{I})$

        Arguments:
            X: Input data of shape `(batch_size, input_dim)`.

        Returns:
            mean: Mean vector of shape `(batch_size, latent_dim)`.
            log_std: Log standard deviation of shape `(batch_size, d_var_enc)`.
        """
        # Reshape in case of images
        if len(X.shape) > 3:
            X = torch.flatten(X, start_dim=1)  # (batch_size, input_dim)

        out = self.encoder(X)  # (batch_size, latent_dim + d_var_enc)

        mean, log_std = (
            out[:, : -self.d_var_enc],
            out[:, -self.d_var_enc :],
        )  # (batch_size, latent_dim), (batch_size, d_var_enc)
        return mean, log_std

    def get_raw_output_dec(
        self, samples: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        Compute the mean and log standard deviation of the likelihood distribution $p(\mathbf{x}|\mathbf{z})$.

        The likelihood distribution is parameterized as:
        $p(\mathbf{x} \mid \mathbf{z}) = \mathcal N (\mathbf{x} \mid \mu(\mathbf{z}), \sigma(\mathbf{z}) \mathbf{I})

        Arguments:
            samples: Samples from the posterior distribution of shape `(batch_size, latent_dim)`.

        Returns:
            mean: Mean of the likelihood of shape `(batch_size, input_dim)`.
            log_std: Log standard deviation of shape `(batch_size, d_var_dec)`.
        """

        # Decode Samples
        out_rec = self.decoder(samples)  # (batch_size, input_dim + d_var_dec)
        # Obtain mean and var
        mean, log_std = (
            out_rec[:, : -self.d_var_dec],
            out_rec[:, -self.d_var_dec :],
        )  # (batch_size, input_dim), (batch_size, d_var_dec)

        if self.covar_mode == "single":
            log_std = torch.ones_like(mean) * log_std

        return mean, log_std

    def forward(
        self, X: torch.Tensor, n_samples: int = 1, return_mean_logstd: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Forward pass through the VAE encoder.

        Note: This method only implements encoding, since the latent variables are used for downstream tasks.

        Arguments:
            X: Input data of shape `(batch_size, ...)`.
            n_samples: Number of Monte Carlo samples to generate from the posterior.
            return_mean_logstd: If True, also returns the posterior mean and log standard deviation.

        Returns:
            posterior_samples: Samples from the posterior $q(\mathbf{z}|\mathbf{x})$ of shape `(batch_size, n_samples, latent_dim)`.
            post_mean: Only returned when `return_mean_logstd=True`. Posterior mean of shape `(batch_size, latent_dim)`.
            post_log_std: Only returned when `return_mean_logstd=True`. Posterior log std of shape `(batch_size, latent_dim)`.
        """

        if len(X.shape) > 3:
            X = torch.flatten(X, start_dim=1)

        posterior_samples, post_mean, post_log_std = self.get_posterior_samples(
            X=X, n_samples=n_samples, return_mean_logstd=True
        )

        if return_mean_logstd:
            return posterior_samples, post_mean, post_log_std

        return posterior_samples

    def get_posterior_samples(
        self, X: torch.Tensor, n_samples: int = 1, return_mean_logstd: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Generate samples from the posterior distribution $q(\mathbf{z}|\mathbf{x})$.

        Arguments:
            X: Input data of shape `(batch_size, input_dim)`.
            n_samples: Number of samples to obtain.
            return_mean_logstd: Whether to return the mean and log_std used to obtain the samples.

        Returns:
            rep_samples: Samples of $q(\mathbf{z}|\mathbf{x})$ of shape `(batch_size, n_samples, latent_dim)`.
            mean: Only returned when `return_mean_logstd=True`. Mean of shape `(batch_size, latent_dim)`.
            log_std_v: Only returned when `return_mean_logstd=True`. Log std of shape `(batch_size, latent_dim)`.
        """
        # Flatten in case of images
        if len(X.shape) > 2:
            X = torch.flatten(X, start_dim=1)

        mean, log_std = self.get_raw_output_enc(
            X
        )  # (batch_size, latent_dim), (batch_size, d_var_enc)

        # Current implementation only stands single variance for all the latent vector
        if self.covar_mode == "single":
            log_std_v = torch.ones_like(mean) * log_std  # (batch_size, latent_dim)
        elif self.covar_mode == "diagonal":
            log_std_v = log_std  # (batch_size, latent_dim)

        rep_samples = self.get_reparameterized_samples(
            mean, log_std_v, n_samples=n_samples
        )  # (batch_size, n_samples, latent_dim)

        if return_mean_logstd:
            return rep_samples, mean, log_std_v

        return rep_samples

    def complete_forward_samples(
        self, X: torch.Tensor, n_samples: int = 1
    ) -> torch.Tensor:
        r"""
        Compute samples from the likelihood $p(\mathbf{x}|\mathbf{z})$ using a complete forward pass.

        This method first encodes the input to obtain latent samples, then decodes
        these samples to reconstruct the original input.

        Arguments:
            X: Input data of shape `(batch_size, input_dim)`.
            n_samples: Number of Monte Carlo samples for the forward pass.

        Returns:
            reconstructions: Reconstructed data of shape `(batch_size, input_dim)`.
        """
        orig_shape = X.shape

        samples, mean, log_std = self.get_posterior_samples(
            X, n_samples=n_samples, return_mean_logstd=True
        )  #  (batch_size, n_samples, latent_dim), (batch_size, latent_dim), (batch_size, latent_dim)

        # Flatten the MC dimension
        samples = samples.reshape(
            samples.shape[0] * samples.shape[1], samples.shape[2]
        )  # (batch_size * n_samples, latent_dim)
        # Compute likelihood mean and log_std
        lik_mean, lik_log_std = self.get_raw_output_dec(
            samples
        )  #  (batch_size * n_samples, input_dim), (batch_size * n_samples, d_var_dec)

        # Reshape back to (batch_size, n_samples, input_dim) and take mean over samples
        lik_mean = lik_mean.view(
            orig_shape[0], n_samples, orig_shape[1]
        )  # (batch_size, n_samples, input_dim)
        lik_mean = lik_mean.mean(
            dim=1
        )  # (batch_size, input_dim) - average over samples

        return lik_mean

    def compute_loss(
        self,
        X: torch.Tensor,
        reduction: str = "mean",
        n_samples: int = 1,
        return_samples: bool = False,
    ) -> dict | tuple[dict, torch.Tensor]:
        r"""
        Compute the ELBO:

        $\mathcal E_q[log p(\mathbf{x}|\mathbf{z})] - KL[q(\mathbf{z})||p(\mathbf{z})]$

        Arguments:
            X: Input data of shape `(batch_size, input_dim)`.
            reduction: Way to reduce the loss across instances. Options: 'sum', 'mean', 'none'.
            n_samples: Number of Monte Carlo samples for the loss computation.
            return_samples: If True, also returns the latent samples used for loss computation.

        Returns:
            loss_dict: Dictionary containing 'VaeELL' (negative expected log-likelihood) and 'VaeKL' (KL divergence).
            samples: Only returned when `return_samples=True`. Latent samples of shape `(batch_size * n_samples, latent_dim)`.
        """

        if len(X.shape) > 2:
            X = torch.flatten(X, start_dim=1)  # (batch_size, ...)

        samples, post_mean, post_log_std = self.get_posterior_samples(
            X, n_samples=n_samples, return_mean_logstd=True
        )  #  (batch_size, n_samples, latent_dim), (batch_size, latent_dim), (batch_size, latent_dim)

        # Flatten the MC dimension
        samples = samples.reshape(
            samples.shape[0] * samples.shape[1], samples.shape[2]
        )  # (batch_size * n_samples, latent_dim)

        # Compute likelihood mean and log_std
        lik_mean, lik_log_std = self.get_raw_output_dec(
            samples
        )  #  (batch_size * n_samples, input_dim), (batch_size * n_samples, d_var_dec)

        # Replicate for each MC sample
        X_replicated = X.repeat_interleave(n_samples, dim=0)

        # Compute for all inputs LL
        ell = self._diagonal_log_gaussian_pdf(
            X_replicated, lik_mean, lik_log_std
        )  # (batch_size * n_samples)

        kl = self._kl_prior(post_mean, post_log_std)  # (batch_size)

        # Reshape the ELL and compute mean in MC samples
        ell = ell.view(X.shape[0], n_samples).mean(
            dim=1
        )  # (batch_size, n_samples) -> (batch_size)

        # Reduce dimensions, change sign to KL
        if reduction == "mean":
            ell = torch.mean(ell)  # ()
            kl = torch.mean(kl)  # ()
        elif reduction == "sum":
            ell = torch.sum(ell)  # ()
            kl = torch.sum(kl)  # ()

        # Care: Current implementation returns -ELL
        if return_samples:
            return {"VaeELL": -ell, "VaeKL": kl}, samples
        return {"VaeELL": -ell, "VaeKL": kl}

    def _kl_prior(self, mean: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
        r"""
        Compute KL divergence between posterior $q(\mathbf{z}|\mathbf{x})$ and standard normal prior.

        Computes $D_{KL}(q_\phi(\mathbf{z}|\mathbf{x}) || \mathcal{N}(0, I))$ for a multivariate Gaussian
        posterior with diagonal covariance matrix.

        Arguments:
            mean: Posterior mean vectors of shape `(batch_size, latent_dim)`.
            log_std: Posterior log standard deviations of shape `(batch_size, latent_dim)`.

        Returns:
            kl_div: KL divergence per instance of shape `(batch_size,)`.
        """

        kl_div = -0.5 * torch.sum(
            1 + 2 * (log_std) - mean**2 - torch.exp(2 * log_std), dim=1
        )

        return kl_div

    def _diagonal_log_gaussian_pdf(
        self, inputs: torch.Tensor, mean: torch.Tensor, log_std: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Compute log probability density of a diagonal Gaussian.

        Computes $\log \mathcal{N}(x; \mu, \sigma^2 I)$ for inputs with diagonal covariance.

        Arguments:
            inputs: Input data of shape `(batch_size, input_dim)`.
            mean: Gaussian mean of shape `(batch_size, input_dim)`.
            log_std: Gaussian log standard deviation of shape `(batch_size, input_dim)`.

        Returns:
            log_prob: Log probability densities of shape `(batch_size,)`.
        """

        # Const
        K = inputs.shape[-1]
        log_std = log_std

        # Const term. Det of Sigma is prod of diag.
        log_det_cov = 2 * torch.sum(log_std, dim=1)

        # Compute terms
        inv_cov = torch.exp(-2 * log_std)
        diff = inputs - mean

        quadratic_term = torch.sum(diff * diff * inv_cov, dim=1)

        # log density
        log_pdf = -0.5 * (
            K * torch.log(torch.tensor(2 * np.pi)) + log_det_cov + quadratic_term
        )

        return log_pdf

    def log_marginal_likelihood_importance_sampling(
        self, X: torch.Tensor, n_samples: int = 1
    ) -> torch.Tensor:
        r"""
        Compute log marginal likelihood log $p(\mathbf{x})$ via importance sampling. The estimation is computed as

        $\log p(\mathbf{x}) \approx \log \frac{1}{K} \sum_{i=1}^K \frac{p(x|z_i)p(z_i)}{q(z_i|x)}$

        Arguments:
            X: Input data of shape `(batch_size, input_dim)`.
            n_samples: Number of importance samples for estimation.

        Returns:
            log_marginal: Log marginal likelihood estimates of shape `(batch_size,)`.
        """
        # Case of bag of images
        if len(X.shape) > 2:
            X = torch.flatten(X, start_dim=1)  # (batch_size,  ...)

        # Encode
        samples, mean_post, log_std_post = self.get_posterior_samples(
            X, n_samples=n_samples, return_mean_logstd=True
        )  # (batch_size, n_samples, latent_dim), (batch_size, latent_dim), (batch_size, latent_dim)
        samples = samples.reshape(
            samples.shape[0] * samples.shape[1], -1
        )  # (batch_size * n_samples, latent_dim)

        # Decode Samples
        mean_lik, log_std_lik = self.get_raw_output_dec(
            samples
        )  #  (batch_size * n_samples, input_dim), (batch_size * n_samples, d_var_dec)

        # Replicate for each MC sample
        X_replicated = X.repeat_interleave(
            n_samples, dim=0
        )  # (batch_size * n_samples, input_dim)
        mean_post = mean_post.repeat_interleave(
            n_samples, dim=0
        )  # (batch_size * n_samples, latent_dim)
        log_std_post = log_std_post.repeat_interleave(
            n_samples, dim=0
        )  # (batch_size * n_samples, latent_dim)

        # Compute marginal LLs
        ll = self._diagonal_log_gaussian_pdf(X_replicated, mean_lik, log_std_lik)
        prior_log_lik = self._diagonal_log_gaussian_pdf(
            samples, torch.zeros_like(samples), torch.ones_like(samples)
        )
        post_log_lik = self._diagonal_log_gaussian_pdf(samples, mean_post, log_std_post)

        # Compute via log sum exp as:
        # log p(x) = log (1/n_samples) * sum_i exp(log p(x|z_i) + log p(z_i) - log q(z_i|x))
        log_exponent = ll + prior_log_lik - post_log_lik  # (batch_size * n_samples)
        log_exponent = log_exponent.view(
            X.shape[0], n_samples
        )  # (batch_size, n_samples)
        unnormalized_log_marginal_imp_sampling = torch.logsumexp(
            log_exponent, dim=1
        )  # (batch_size)
        log_marginal_importance = unnormalized_log_marginal_imp_sampling + torch.log(
            torch.tensor(1 / n_samples, dtype=torch.float64, device=X.device)
        )

        return log_marginal_importance


class VariationalAutoEncoderMIL(VariationalAutoEncoder):
    r"""
    Variational Autoencoder for Multiple Instance Learning.

    This class extends the VAE to handle bag-structured data by processing each instance
    in a bag independently through the VAE and returning results in bag format.
    """

    def forward(
        self, X: torch.Tensor, n_samples: int = 1, return_mean_logstd: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for bag-structured data.

        This method processes each instance in the bags independently through the VAE encoder.
        Used in MIL feature extraction where output must be `(batch_size, bag_size, latent_dim)`.

        Arguments:
            X: Bag data of shape `(batch_size, bag_size, input_dim)`.
            n_samples: Number of Monte Carlo samples.
            return_mean_logstd: Whether to return posterior mean and log standard deviation.

        Returns:
            samples: Encoding samples of shape `(batch_size, bag_size, n_samples, latent_dim)`.
            mean: Only returned when `return_mean_logstd=True`. Mean of shape `(batch_size, bag_size, latent_dim)`.
            log_std: Only returned when `return_mean_logstd=True`. Log std of shape `(batch_size, bag_size, latent_dim)`.
        """
        if len(X.shape) == 2:
            X = X.unsqueeze(0)

        B, N = X.shape[0], X.shape[1]

        # View as individual elements and forward
        samples, mean, log_std = super().forward(
            X.view(B * N, *X.shape[2:]), n_samples, return_mean_logstd=True
        )  # (batch_size * bag_size, n_samples, latent_dim), # (batch_size * bag_size, latent_dim), (batch_size * bag_size, latent_dim)

        samples = samples.view(
            B, N, n_samples, mean.shape[-1]
        )  # (batch_size, bag_size, n_samples, latent_dim)

        if return_mean_logstd:
            mean = mean.view(B, N, mean.shape[-1])  # (batch_size, bag_size, latent_dim)
            log_std = log_std.view(
                B, N, log_std.shape[-1]
            )  # (batch_size, bag_size, latent_dim)
            return samples, mean, log_std

        return samples

    def log_marginal_likelihood_importance_sampling(
        self, X: torch.Tensor, mask: torch.Tensor | None = None, n_samples: int = 1
    ) -> torch.Tensor:
        """
        Compute log marginal likelihood for bag-structured data via importance sampling.

        This method processes each instance in the bags independently and returns
        log marginal estimates for each instance.

        Arguments:
            X: Bag data of shape `(batch_size, bag_size, input_dim)`.
            mask: Optional binary mask of shape `(batch_size, bag_size)` indicating valid instances.
            n_samples: Number of importance samples for estimation.

        Returns:
            log_marginal: Log marginal likelihood per instance of shape `(batch_size, bag_size)`.
        """
        B, N = X.shape[0], X.shape[1]
        mask = mask if mask is not None else torch.ones(B, N).to(X.device)
        log_marginal_imp = super().log_marginal_likelihood_importance_sampling(
            X.view(B * N, *X.shape[2:]), n_samples
        )  # (batch_size * bag_size)
        return log_marginal_imp.view(B, N) * mask

    def compute_loss(
        self,
        X: torch.Tensor,
        mask: torch.Tensor | None = None,
        reduction: str = "mean",
        n_samples: int = 1,
        return_samples: bool = False,
    ) -> dict | tuple[dict, torch.Tensor]:
        """
        Compute VAE loss for bag-structured data.

        The loss is computed for each instance in the bags and then aggregated according
        to the reduction strategy and optional mask.

        Arguments:
            X: Bag data of shape `(batch_size, bag_size, input_dim)`.
            mask: Optional binary mask of shape `(batch_size, bag_size)` for valid instances.
            reduction: Reduction method ('sum', 'mean', or 'none').
            n_samples: Number of Monte Carlo samples for loss computation.
            return_samples: Whether to return latent samples used in loss computation.

        Returns:
            loss_dict: Dictionary with 'VaeELL' and 'VaeKL' losses.
            samples: Only returned when `return_samples=True`. Latent samples used for loss computation.
        """

        B, N, _ = X.shape[0], X.shape[1], X.shape[2:]
        mask = mask if mask is not None else torch.ones(B, N)

        X = X.view(B * N, *X.shape[2:])

        # Recall that Super returns {-ell, KL}
        losses, samples = super().compute_loss(
            X, reduction="none", n_samples=n_samples, return_samples=True
        )

        # Assumes that the MC samples are reduced in the VAE
        losses["VaeELL"] = losses["VaeELL"].view(B, N)
        losses["VaeKL"] = losses["VaeKL"].view(B, N)

        if reduction == "sum":
            losses["VaeELL"] = torch.sum(losses["VaeELL"].sum(dim=-1))
            losses["VaeKL"] = torch.sum(losses["VaeKL"].sum(dim=-1))
        elif reduction == "mean":
            losses["VaeELL"] = torch.mean(losses["VaeELL"].mean(dim=-1))
            losses["VaeKL"] = torch.mean(losses["VaeKL"].mean(dim=-1))

        if return_samples:
            return losses, samples.view(B, n_samples, N, -1)
        return losses

    def complete_forward_samples(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute reconstructions for bag-structured data via complete forward pass.

        This method processes each instance in the bags independently through the
        VAE encoder-decoder pipeline and returns reconstructions in bag format.

        Arguments:
            X: Bag data of shape `(batch_size, bag_size, input_dim)`.

        Returns:
            reconstructions: Reconstructed bag data of shape `(batch_size, bag_size, input_dim)`.
        """
        B, N, D = X.shape[0], X.shape[1], *X.shape[2:]
        recs = super().complete_forward_samples(
            X.view(B * N, *X.shape[2:])
        )  # (batch_size * bag_size * n_samples)
        return recs.view(B, N, D)
