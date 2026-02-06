import torch
import pytest

from torchmil.models import VAEABMIL  # Import the VAEABMIL class
from torchmil.nn import VariationalAutoEncoderMIL


# Fixtures for common setup
@pytest.fixture
def sample_data():
    # Returns a tuple of (X, Y, mask)
    torch.manual_seed(42)  # For reproducibility
    X = torch.randn(2, 3, 10)  # batch_size, bag_size, feat_dim
    Y = torch.randint(0, 2, (2,)).float()  # batch_size, ensure float for BCE loss
    mask = torch.ones(2, 3).bool()  # All instances are valid for more stable testing
    return X, Y, mask


@pytest.fixture
def vae_feat_ext():
    # Returns a VariationalAutoEncoderMIL instance for feature extraction
    return VariationalAutoEncoderMIL(
        input_shape=(10,), layer_sizes=[8, 5], activations=["relu", "None"]
    )


@pytest.fixture
def vaeabmil_model(vae_feat_ext):
    # Returns an instance of the VAEABMIL model with default parameters
    return VAEABMIL(feat_ext=vae_feat_ext, in_shape=(3, 10))


# Basic tests for VAEABMIL class
def test_vaeabmil_initialization(vae_feat_ext):
    # Test basic initialization
    model = VAEABMIL(feat_ext=vae_feat_ext, in_shape=(3, 10))
    assert model is not None


def test_vaeabmil_forward_pass(sample_data, vaeabmil_model):
    # Test basic forward pass
    X, _, mask = sample_data

    Y_pred = vaeabmil_model(X, mask)
    assert Y_pred.shape == (2,)


def test_vaeabmil_forward_with_attention(sample_data, vaeabmil_model):
    # Test forward pass with attention return
    X, _, mask = sample_data

    Y_pred, att = vaeabmil_model(X, mask, return_att=True)
    assert Y_pred.shape == (2,)
    assert att.shape == (2, 3)


def test_vaeabmil_compute_loss(sample_data, vaeabmil_model):
    # Test loss computation
    X, Y, mask = sample_data

    # Ensure the model is in a good state for loss computation
    with torch.no_grad():
        _ = vaeabmil_model(X, mask)

    Y_pred, loss_dict = vaeabmil_model.compute_loss(Y, X, mask)

    assert Y_pred.shape == (2,)
    assert "BCEWithLogitsLoss" in loss_dict
    assert "VaeELL" in loss_dict
    assert "VaeKL" in loss_dict


def test_vaeabmil_predict(sample_data, vaeabmil_model):
    # Test predict method
    X, _, mask = sample_data

    Y_pred = vaeabmil_model.predict(X, mask, return_inst_pred=False)
    assert Y_pred.shape == (2,)

    Y_pred, y_inst_pred = vaeabmil_model.predict(X, mask, return_inst_pred=True)
    assert Y_pred.shape == (2,)
    assert y_inst_pred.shape == (2, 3)
