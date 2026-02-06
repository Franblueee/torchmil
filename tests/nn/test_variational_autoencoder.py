import torch
import pytest

from torchmil.nn.variational_autoencoder import (
    VariationalAutoEncoder,
    VariationalAutoEncoderMIL,
)

# --- Fixtures ---

@pytest.fixture
def input_dim():
    return 10

@pytest.fixture
def latent_dim():
    return 5

@pytest.fixture
def layer_sizes(latent_dim):
    return [8, latent_dim]

@pytest.fixture
def sample_data(input_dim):
    return torch.randn(4, input_dim)  # batch_size=4, input_dim=10

@pytest.fixture
def sample_image_data():
    # Batch=2, Channels=3, H=4, W=4 -> Flattened size = 48
    return torch.randn(2, 3, 4, 4)

@pytest.fixture
def sample_bag_data(input_dim):
    return torch.randn(2, 3, input_dim)  # batch_size=2, bag_size=3, input_dim=10

@pytest.fixture
def vae_basic(input_dim, layer_sizes):
    return VariationalAutoEncoder(
        input_shape=(input_dim,), 
        layer_sizes=layer_sizes, 
        activations=["relu", "None"],
        covar_mode="single"
    )

@pytest.fixture
def vae_diagonal(input_dim, layer_sizes):
    return VariationalAutoEncoder(
        input_shape=(input_dim,), 
        layer_sizes=layer_sizes, 
        covar_mode="diagonal"
    )

@pytest.fixture
def vae_mil(input_dim, layer_sizes):
    return VariationalAutoEncoderMIL(
        input_shape=(input_dim,), 
        layer_sizes=layer_sizes, 
        activations=["relu", "None"]
    )

# --- Tests for VariationalAutoEncoder (Standard) ---

def test_vae_init_validations():
    """Test initialization logic and error handling."""
    # Test valid diagonal init
    vae = VariationalAutoEncoder(input_shape=(10,), layer_sizes=[5], covar_mode="diagonal")
    assert vae.d_var_enc == 5  # Last layer size
    assert vae.d_var_dec == 10 # Input shape
    
    # Test invalid covar mode
    with pytest.raises(NotImplementedError, match="not valid"):
        VariationalAutoEncoder(input_shape=(10,), covar_mode="invalid_mode")

def test_vae_forward_standard(sample_data, vae_basic, latent_dim):
    """Test standard forward pass and shape."""
    samples = vae_basic(sample_data, n_samples=2)
    assert samples.shape == (4, 2, latent_dim)

def test_vae_forward_image_input(sample_image_data):
    """
    Test the flattening logic in forward/get_posterior_samples.
    The input is (B, C, H, W). The VAE must be init with flat dim size.
    """
    flat_dim = 3 * 4 * 4
    vae = VariationalAutoEncoder(input_shape=(flat_dim,), layer_sizes=[10])
    
    # This hits the `if len(X.shape) > 3` block
    samples = vae(sample_image_data, n_samples=1)
    assert samples.shape == (2, 1, 10)

def test_vae_forward_returns_stats(sample_data, vae_basic, latent_dim):
    """Test forward with return_mean_logstd=True."""
    samples, mean, log_std = vae_basic(sample_data, n_samples=1, return_mean_logstd=True)
    assert samples.shape == (4, 1, latent_dim)
    assert mean.shape == (4, latent_dim)
    # In 'single' mode, log_std returned by forward is expanded to match mean shape logic?
    # Looking at code: `log_std_v = torch.ones_like(mean) * log_std`
    assert log_std.shape == (4, latent_dim) 

def test_vae_diagonal_covariance_logic(sample_data, vae_diagonal, input_dim, latent_dim):
    """
    Test flow specifically for diagonal covariance mode.
    Verifies dimensions of variances in encoder and decoder.
    """
    # 1. Encoder Raw Output
    mean, log_std = vae_diagonal.get_raw_output_enc(sample_data)
    assert mean.shape == (4, latent_dim)
    assert log_std.shape == (4, latent_dim) # Diagonal mode: var dim == latent dim
    
    # 2. Decoder Raw Output
    latent_sample = torch.randn(4, latent_dim)
    dec_mean, dec_log_std = vae_diagonal.get_raw_output_dec(latent_sample)
    assert dec_mean.shape == (4, input_dim)
    assert dec_log_std.shape == (4, input_dim) # Diagonal mode: var dim == input dim

def test_vae_complete_forward_samples(sample_data, vae_basic):
    """Test reconstruction path."""
    recs = vae_basic.complete_forward_samples(sample_data, n_samples=5)
    # Result is averaged over samples
    assert recs.shape == sample_data.shape

def test_vae_compute_loss_variants(sample_data, vae_basic):
    """Test all reduction modes and return flags in compute_loss."""
    # 1. Reduction = Sum
    loss_sum = vae_basic.compute_loss(sample_data, reduction="sum")
    assert loss_sum["VaeELL"].ndim == 0
    
    # 2. Reduction = None (returns per instance)
    loss_none = vae_basic.compute_loss(sample_data, reduction="none")
    assert loss_none["VaeELL"].shape == (4,)
    assert loss_none["VaeKL"].shape == (4,)
    
    # 3. Return Samples
    loss_dict, samples = vae_basic.compute_loss(sample_data, return_samples=True)
    assert "VaeELL" in loss_dict
    assert samples.shape[0] == 4 * 1 # Batch * n_samples

def test_vae_compute_loss_image_flattening(sample_image_data):
    """Test that compute_loss handles >2D input (flattening)."""
    flat_dim = 3 * 4 * 4
    vae = VariationalAutoEncoder(input_shape=(flat_dim,), layer_sizes=[10])
    # This hits `if len(X.shape) > 2` inside compute_loss
    loss = vae.compute_loss(sample_image_data)
    assert "VaeELL" in loss

def test_vae_importance_sampling(sample_data, vae_basic):
    """Test log_marginal_likelihood_importance_sampling."""
    log_imp = vae_basic.log_marginal_likelihood_importance_sampling(sample_data, n_samples=10)
    assert log_imp.shape == (4,)
    
    # Test with image data (flattening check)
    flat_dim = 3 * 4 * 4
    img_data = torch.randn(2, 3, 4, 4)
    vae_img = VariationalAutoEncoder(input_shape=(flat_dim,), layer_sizes=[10])
    log_imp_img = vae_img.log_marginal_likelihood_importance_sampling(img_data, n_samples=2)
    assert log_imp_img.shape == (2,)

# --- Tests for VariationalAutoEncoderMIL (MIL Extension) ---

def test_mil_forward_structure(sample_bag_data, vae_mil, latent_dim):
    """Test basic MIL forward pass dimensions."""
    # Input: (2, 3, 10) -> Output: (2, 3, n_samples, latent)
    samples = vae_mil(sample_bag_data, n_samples=2)
    assert samples.shape == (2, 3, 2, latent_dim)

def test_mil_forward_single_instance_edge_case(input_dim, vae_mil):
    """Test forward pass when input is (BagSize, Dim) instead of (Batch, Bag, Dim)."""
    single_bag = torch.randn(5, input_dim) 
    # The code `if len(X.shape) == 2: X = X.unsqueeze(0)` handles this
    samples = vae_mil(single_bag, n_samples=1)
    # Output should be (1, 5, 1, latent)
    assert samples.shape == (1, 5, 1, vae_mil.layer_sizes[-1])

def test_mil_forward_return_stats(sample_bag_data, vae_mil, latent_dim):
    """Test return_mean_logstd in MIL context."""
    samples, mean, log_std = vae_mil(sample_bag_data, n_samples=1, return_mean_logstd=True)
    assert mean.shape == (2, 3, latent_dim)
    assert log_std.shape == (2, 3, latent_dim)

def test_mil_complete_forward(sample_bag_data, vae_mil):
    """Test reconstruction in MIL context."""
    recs = vae_mil.complete_forward_samples(sample_bag_data)
    assert recs.shape == sample_bag_data.shape

def test_mil_compute_loss_masking(sample_bag_data, vae_mil):
    """Test loss computation with and without masks, and different reductions."""
    # Mask: 1 for valid, 0 for padding. Let's mask the last instance of bag 0.
    mask = torch.ones(2, 3)
    mask[0, 2] = 0 
    
    # 1. Reduction Mean
    loss_mean = vae_mil.compute_loss(sample_bag_data, mask=mask, reduction="mean")
    assert isinstance(loss_mean["VaeELL"], torch.Tensor)
    
    # 2. Reduction Sum
    loss_sum = vae_mil.compute_loss(sample_bag_data, mask=mask, reduction="sum")
    assert isinstance(loss_sum["VaeELL"], torch.Tensor)
    
    # 3. Reduction None (should return grid)
    loss_none = vae_mil.compute_loss(sample_bag_data, mask=mask, reduction="none")
    assert loss_none["VaeELL"].shape == (2, 3)
    
    # 4. Return Samples
    loss_dict, samples = vae_mil.compute_loss(sample_bag_data, return_samples=True)
    # Expected sample shape: (Batch, n_samples, BagSize, Latent) 
    # Note: Code returns `samples.view(B, n_samples, N, -1)`
    assert samples.shape == (2, 1, 3, vae_mil.layer_sizes[-1])

def test_mil_importance_sampling(sample_bag_data, vae_mil):
    """Test MIL importance sampling with mask."""
    mask = torch.ones(2, 3)
    log_imp = vae_mil.log_marginal_likelihood_importance_sampling(
        sample_bag_data, mask=mask, n_samples=5
    )
    assert log_imp.shape == (2, 3) # Returns (Batch, BagSize)
    
    # Test without mask (defaults to ones)
    log_imp_nomask = vae_mil.log_marginal_likelihood_importance_sampling(
        sample_bag_data, n_samples=5
    )
    assert log_imp_nomask.shape == (2, 3)