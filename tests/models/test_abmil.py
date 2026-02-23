import torch
import pytest
from torch import nn

from torchmil.models import ABMIL  # Import the ABMIL class


# Fixtures for common setup
@pytest.fixture
def sample_data():
    # Returns a tuple of (X, Y, mask)
    X = torch.randn(2, 3, 5)  # batch_size, bag_size, feat_dim
    Y = torch.randint(0, 2, (2,))  # batch_size
    mask = torch.randint(0, 2, (2, 3)).bool()  # batch_size, bag_size
    return X, Y, mask


@pytest.fixture
def abmil_model():
    # Returns an instance of the ABMIL model with default parameters
    return ABMIL(in_shape=(3, 5))


# Tests for ABMIL class
def test_abmil_initialization():
    # Tests that the model can be initialized with different parameters
    ABMIL(in_shape=(3, 5))
    ABMIL(in_shape=(3, 5), att_dim=64)
    ABMIL(in_shape=(3, 5), att_act="relu")
    ABMIL(in_shape=(3, 5), gated=True)
    ABMIL(in_shape=(3, 5), feat_ext=nn.Linear(5, 10))
    ABMIL(in_shape=(3, 5), criterion=nn.MSELoss())


def test_abmil_forward_pass(sample_data, abmil_model):
    # Tests the forward pass of the model with and without mask and return_att
    X, _, mask = sample_data
    Y_pred = abmil_model(X, mask)
    assert Y_pred.shape == (2,)

    Y_pred = abmil_model(X)
    assert Y_pred.shape == (2,)

    Y_pred, att = abmil_model(X, mask, return_att=True)
    assert Y_pred.shape == (2,)
    assert att.shape == (2, 3)


def test_abmil_compute_loss(sample_data, abmil_model):
    # Tests the compute_loss method of the model
    X, Y, mask = sample_data
    Y_pred, loss_dict = abmil_model.compute_loss(Y, X, mask)
    assert Y_pred.shape == (2,)
    assert "BCEWithLogitsLoss" in loss_dict
    assert loss_dict["BCEWithLogitsLoss"].shape == ()  # loss is a scalar


def test_abmil_predict(sample_data, abmil_model):
    # Tests the predict method of the model with and without return_inst_pred
    X, _, mask = sample_data
    Y_pred = abmil_model.predict(X, mask, return_inst_pred=False)
    assert Y_pred.shape == (2,)

    Y_pred, y_inst_pred = abmil_model.predict(X, mask, return_inst_pred=True)
    assert Y_pred.shape == (2,)
    assert y_inst_pred.shape == (2, 3)


def test_abmil_with_feature_extractor(sample_data):
    # Tests the model with a feature extractor
    X, Y, mask = sample_data
    feat_ext = nn.Sequential(
        nn.Linear(5, 10),
        nn.ReLU(),
        nn.Linear(10, 7),
    )
    model = ABMIL(
        in_shape=(3, 5), feat_ext=feat_ext
    )  # in_shape is the shape of the original input
    Y_pred = model(X, mask)
    assert Y_pred.shape == (2,)

    Y_pred, loss_dict = model.compute_loss(Y, X, mask)
    assert Y_pred.shape == (2,)
    assert "BCEWithLogitsLoss" in loss_dict
    assert loss_dict["BCEWithLogitsLoss"].shape == ()


def test_abmil_no_in_shape(sample_data):
    # Test case where in_shape is not provided during initialization.
    X, Y, mask = sample_data
    model = ABMIL()  # Initialize without in_shape
    # The model should be able to infer the input shape during the forward pass.
    Y_pred = model(X, mask)
    assert Y_pred.shape == (2,)
    Y_pred, loss_dict = model.compute_loss(Y, X, mask)
    assert Y_pred.shape == (2,)
    assert "BCEWithLogitsLoss" in loss_dict
    assert loss_dict["BCEWithLogitsLoss"].shape == ()


def test_abmil_different_pooling_params(sample_data):
    # Test different attention pooling parameters
    X, Y, mask = sample_data
    model_relu = ABMIL(in_shape=(3, 5), att_act="relu")
    model_gelu = ABMIL(in_shape=(3, 5), att_act="gelu")
    model_gated = ABMIL(in_shape=(3, 5), gated=True)

    Y_pred_relu = model_relu(X, mask)
    Y_pred_gelu = model_gelu(X, mask)
    Y_pred_gated = model_gated(X, mask)

    assert Y_pred_relu.shape == (2,)
    assert Y_pred_gelu.shape == (2,)
    assert Y_pred_gated.shape == (2,)


def test_abmil_multiclass_support(sample_data):
    # Tests multiclass support with different numbers of outputs
    X, _, mask = sample_data

    # Test binary classification (default)
    model_binary = ABMIL(in_shape=(3, 5), n_outputs=1)
    assert model_binary.num_outputs == 1
    Y_pred = model_binary(X, mask)
    assert Y_pred.shape == (2,)

    # Test multiclass with 3 classes
    model_3class = ABMIL(in_shape=(3, 5), n_outputs=3)
    assert model_3class.num_outputs == 3
    Y_pred = model_3class(X, mask)
    assert Y_pred.shape == (2, 3)

    # Test multiclass with 5 classes
    model_5class = ABMIL(in_shape=(3, 5), n_outputs=5)
    assert model_5class.num_outputs == 5
    Y_pred = model_5class(X, mask)
    assert Y_pred.shape == (2, 5)
