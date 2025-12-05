# Variational Autoencoder
::: torchmil.nn.VariationalAutoEncoder
    options:
        members:
        - __init__
        - get_reparameterized_samples
        - get_raw_output_enc
        - get_raw_output_dec
        - forward
        - get_posterior_samples
        - complete_forward_samples
        - compute_loss
        - _kl_prior
        - _diagonal_log_gaussian_pdf
        - log_marginal_likelihood_importance_sampling
-------------------------
::: torchmil.nn.VariationalAutoEncoderMIL
    options:
        members:
        - __init__
        - forward
        - log_marginal_likelihood_importance_sampling
        - compute_loss
        - complete_forward_samples
