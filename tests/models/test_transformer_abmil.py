import torch
import pytest
from torch import nn

from torchmil.models import TransformerABMIL  # Import the TransformerABMIL class


# Fixtures for common setup
@pytest.fixture
def sample_data():
    # Returns a tuple of (X, Y, mask)
    X = torch.randn(2, 3, 5)  # batch_size, bag_size, feat_dim
    Y = torch.randint(0, 2, (2,))  # batch_size
    mask = torch.randint(0, 2, (2, 3)).bool()  # batch_size, bag_size
    return X, Y, mask


@pytest.fixture
def transformer_abmil_model():
    # Returns an instance of the TransformerABMIL model with default parameters
    return TransformerABMIL(in_shape=(3, 5))


# Tests for TransformerABMIL class
def test_transformer_abmil_initialization():
    # Tests that the model can be initialized with different parameters
    TransformerABMIL(in_shape=(3, 5))
    TransformerABMIL(in_shape=(3, 5), pool_att_dim=64)
    TransformerABMIL(in_shape=(3, 5), pool_act="relu")
    TransformerABMIL(in_shape=(3, 5), pool_gated=True)
    TransformerABMIL(in_shape=(3, 5), feat_ext=nn.Linear(5, 10))
    TransformerABMIL(in_shape=(3, 5), criterion=nn.MSELoss())
    TransformerABMIL(in_shape=(3, 5), transf_att_dim=256)
    TransformerABMIL(in_shape=(3, 5), transf_n_layers=2)
    TransformerABMIL(in_shape=(3, 5), transf_n_heads=4)
    TransformerABMIL(in_shape=(3, 5), transf_use_mlp=False)
    TransformerABMIL(in_shape=(3, 5), transf_add_self=False)
    TransformerABMIL(in_shape=(3, 5), transf_dropout=0.1)


def test_transformer_abmil_forward_pass(sample_data, transformer_abmil_model):
    # Tests the forward pass of the model with and without mask and return_att
    X, _, mask = sample_data
    Y_pred = transformer_abmil_model(X, mask)
    assert Y_pred.shape == (2,)

    Y_pred = transformer_abmil_model(X, mask)
    assert Y_pred.shape == (2,)

    Y_pred, att = transformer_abmil_model(X, mask, return_att=True)
    assert Y_pred.shape == (2,)
    assert att.shape == (2, 3)


def test_transformer_abmil_compute_loss(sample_data, transformer_abmil_model):
    # Tests the compute_loss method of the model
    X, Y, mask = sample_data
    Y_pred, loss_dict = transformer_abmil_model.compute_loss(Y, X, mask)
    assert Y_pred.shape == (2,)
    assert "BCEWithLogitsLoss" in loss_dict
    assert loss_dict["BCEWithLogitsLoss"].shape == ()


def test_transformer_abmil_predict(sample_data, transformer_abmil_model):
    # Tests the predict method of the model with and without return_inst_pred
    X, _, mask = sample_data
    Y_pred = transformer_abmil_model.predict(X, mask, return_inst_pred=False)
    assert Y_pred.shape == (2,)

    Y_pred, y_inst_pred = transformer_abmil_model.predict(
        X, mask, return_inst_pred=True
    )
    assert Y_pred.shape == (2,)
    assert y_inst_pred.shape == (2, 3)


def test_transformer_abmil_with_feature_extractor(sample_data):
    # Tests the model with a feature extractor
    X, Y, mask = sample_data
    feat_ext = nn.Sequential(
        nn.Linear(5, 10),
        nn.ReLU(),
        nn.Linear(10, 7),
    )
    model = TransformerABMIL(in_shape=(3, 5), feat_ext=feat_ext)
    Y_pred = model(X, mask)
    assert Y_pred.shape == (2,)

    Y_pred, loss_dict = model.compute_loss(Y, X, mask)
    assert Y_pred.shape == (2,)
    assert "BCEWithLogitsLoss" in loss_dict
    assert loss_dict["BCEWithLogitsLoss"].shape == ()


def test_transformer_abmil_multiclass_support(sample_data):
    """Test multiclass support for TransformerABMIL model."""
    X, _, mask = sample_data

    # Binary classification (n_outputs=1)
    model_binary = TransformerABMIL(in_shape=(3, 5), n_outputs=1)
    assert model_binary.num_outputs == 1
    Y_pred = model_binary(X, mask)
    assert Y_pred.shape == (2,)

    # 3-class classification (n_outputs=3)
    model_3class = TransformerABMIL(in_shape=(3, 5), n_outputs=3)
    assert model_3class.num_outputs == 3
    Y_pred = model_3class(X, mask)
    assert Y_pred.shape == (2, 3)

    # 5-class classification (n_outputs=5)
    model_5class = TransformerABMIL(in_shape=(3, 5), n_outputs=5)
    assert model_5class.num_outputs == 5
    Y_pred = model_5class(X, mask)
    assert Y_pred.shape == (2, 5)
