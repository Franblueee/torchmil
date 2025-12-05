import torch
import pytest

from torchmil.nn.variational_autoencoder import VariationalAutoEncoder, VariationalAutoEncoderMIL


# Fixtures for common setup
@pytest.fixture
def sample_data():
    return torch.randn(4, 10)  # batch_size=4, input_dim=10


@pytest.fixture
def sample_bag_data():
    return torch.randn(2, 3, 10)  # batch_size=2, bag_size=3, input_dim=10


@pytest.fixture
def vae_basic():
    return VariationalAutoEncoder(
        input_shape=(10,),
        layer_sizes=[8, 5],
        activations=["relu", "None"]
    )


@pytest.fixture
def vae_mil():
    return VariationalAutoEncoderMIL(
        input_shape=(10,),
        layer_sizes=[8, 5],
        activations=["relu", "None"]
    )


# Basic tests for VariationalAutoEncoder class
def test_vae_initialization():
    # Test basic initialization
    vae = VariationalAutoEncoder(
        input_shape=(10,),
        layer_sizes=[8, 5],
        activations=["relu", "None"]
    )
    assert vae.input_dim == (10,)
    assert vae.output_size == 10
    assert vae.layer_sizes == [8, 5]


def test_vae_initialization_diagonal_covar():
    # Test initialization with diagonal covariance
    vae = VariationalAutoEncoder(
        input_shape=(10,),
        layer_sizes=[8, 5],
        covar_mode="diagonal"
    )
    assert vae.covar_mode == "diagonal"


def test_vae_initialization_invalid_covar():
    # Test that invalid covariance mode raises error
    with pytest.raises(NotImplementedError):
        VariationalAutoEncoder(
            input_shape=(10,),
            layer_sizes=[8, 5],
            covar_mode="invalid"
        )


def test_vae_forward(sample_data, vae_basic):
    # Test forward pass (encoding only)
    samples = vae_basic(sample_data, n_samples=2)
    assert samples.shape == (4, 2, 5)  # batch_size, n_samples, latent_dim


def test_vae_get_posterior_samples(sample_data, vae_basic):
    # Test posterior sampling
    samples = vae_basic.get_posterior_samples(sample_data, n_samples=2)
    assert samples.shape == (4, 2, 5)  # batch_size, n_samples, latent_dim


def test_vae_complete_forward_samples(sample_data, vae_basic):
    # Test complete forward pass (encode + decode)
    reconstructions = vae_basic.complete_forward_samples(sample_data, n_samples=1)
    assert reconstructions.shape == sample_data.shape


def test_vae_compute_loss(sample_data, vae_basic):
    # Test loss computation
    loss_dict = vae_basic.compute_loss(sample_data, reduction="sum", n_samples=2)
    
    assert "VaeELL" in loss_dict
    assert "VaeKL" in loss_dict
    assert loss_dict["VaeELL"].shape == ()  # scalar
    assert loss_dict["VaeKL"].shape == ()  # scalar


def test_vae_get_raw_output_enc(sample_data, vae_basic):
    # Test encoder raw output
    mean, log_std = vae_basic.get_raw_output_enc(sample_data)
    
    assert mean.shape == (4, 5)  # batch_size, latent_dim
    assert log_std.shape == (4, 1)  # batch_size, d_var_enc (single mode)


def test_vae_get_raw_output_dec(vae_basic):
    # Test decoder raw output
    latent_samples = torch.randn(4, 5)  # batch_size, latent_dim
    mean, log_std = vae_basic.get_raw_output_dec(latent_samples)
    
    assert mean.shape == (4, 10)  # batch_size, input_dim
    # In single covar mode, log_std is expanded to match input dim
    assert log_std.shape == (4, 10)  # batch_size, input_dim (expanded from d_var_dec)


# Basic tests for VariationalAutoEncoderMIL class
def test_vae_mil_initialization():
    # Test MIL VAE initialization
    vae_mil = VariationalAutoEncoderMIL(
        input_shape=(10,),
        layer_sizes=[8, 5]
    )
    assert isinstance(vae_mil, VariationalAutoEncoder)


def test_vae_mil_forward(sample_bag_data, vae_mil):
    # Test MIL VAE forward pass
    samples = vae_mil(sample_bag_data, n_samples=2)
    assert samples.shape == (2, 3, 2, 5)  # batch_size, bag_size, n_samples, latent_dim


def test_vae_mil_compute_loss(sample_bag_data, vae_mil):
    # Test MIL VAE loss computation
    loss_dict = vae_mil.compute_loss(sample_bag_data, reduction="mean")
    
    assert "VaeELL" in loss_dict and "VaeKL" in loss_dict
    assert loss_dict["VaeELL"].shape == ()
    assert loss_dict["VaeKL"].shape == ()


def test_vae_mil_complete_forward_samples(sample_bag_data, vae_mil):
    # Test complete forward pass for MIL VAE
    reconstructions = vae_mil.complete_forward_samples(sample_bag_data)
    assert reconstructions.shape == sample_bag_data.shape