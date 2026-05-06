import torch
import torch.nn as nn
import pytest

from torchmil.models import DSMIL  # Import the DSMIL class


@pytest.fixture
def sample_input():
    # Example input tensor
    return torch.randn(2, 10, 5)  # batch_size=2, bag_size=10, feature_dim=5


@pytest.fixture
def sample_mask():
    # Example mask tensor
    return torch.randint(0, 2, (2, 10))  # binary mask for batch_size=2, bag_size=10


@pytest.fixture
def sample_labels():
    # Example labels tensor
    return torch.randint(0, 2, (2,))  # binary labels for batch_size = 2


@pytest.fixture
def dsmil_model():
    # Instantiate the DSMIL model with some default parameters.  Good practice to test with different params
    return DSMIL(
        in_shape=(10, 5), att_dim=128, nonlinear_q=False, nonlinear_v=False, dropout=0.0
    )


# Test the forward pass of the DSMIL model
def test_forward_pass(dsmil_model, sample_input, sample_mask):
    Y_pred = dsmil_model(sample_input, sample_mask)
    assert Y_pred.shape == (2,), "Output shape should be (batch_size,)"


def test_forward_pass_with_attention(dsmil_model, sample_input, sample_mask):
    Y_pred, A = dsmil_model(sample_input, sample_mask, return_att=True)
    assert Y_pred.shape == (2,), "Output shape should be (batch_size,)"
    assert A.shape == (2, 10, 1), "Attention shape should be (batch_size, bag_size, 1)"


def test_forward_pass_with_instance_predictions(dsmil_model, sample_input, sample_mask):
    Y_pred, y_logits = dsmil_model(sample_input, sample_mask, return_inst_pred=True)
    assert Y_pred.shape == (2,), "Output shape should be (batch_size,)"
    assert y_logits.shape == (
        2,
        10,
    ), "Instance predictions shape should be (batch_size, bag_size)"


def test_compute_loss(dsmil_model, sample_input, sample_mask, sample_labels):
    Y_pred, loss_dict = dsmil_model.compute_loss(
        sample_labels, sample_input, sample_mask
    )
    assert Y_pred.shape == (2,), "Output shape should be (batch_size,)"
    assert isinstance(loss_dict, dict), "Loss should be returned as a dictionary"
    assert "BCEWithLogitsLoss" in loss_dict, "Dictionary should contain the loss"


def test_predict(dsmil_model, sample_input, sample_mask):
    Y_pred, y_logits_pred = dsmil_model.predict(
        sample_input, sample_mask, return_inst_pred=True
    )
    assert Y_pred.shape == (2,), "Output shape should be (batch_size,)"


def test_predict_with_instance_predictions(dsmil_model, sample_input, sample_mask):
    Y_pred, y_inst_pred = dsmil_model.predict(
        sample_input, sample_mask, return_inst_pred=True
    )
    assert Y_pred.shape == (2,), "Output shape should be (batch_size,)"
    assert y_inst_pred.shape == (
        2,
        10,
    ), "Instance predictions shape should be (batch_size, bag_size)"


# Test with a different feature extractor
def test_with_feature_extractor():
    feat_ext = nn.Sequential(nn.Linear(5, 20), nn.ReLU(), nn.Linear(20, 10))
    model = DSMIL(
        in_shape=(10, 5), feat_ext=feat_ext
    )  # in_shape is the shape *before* the feature extractor.
    sample_input = torch.randn(2, 10, 5)
    sample_mask = torch.randint(0, 2, (2, 10))
    Y_pred = model(sample_input, sample_mask)
    assert Y_pred.shape == (2,)


def test_with_different_criterion():
    criterion = nn.MSELoss()
    model = DSMIL(in_shape=(10, 5), criterion=criterion)
    sample_input = torch.randn(2, 10, 5)
    sample_mask = torch.randint(0, 2, (2, 10))
    sample_labels = torch.randn(2)  # MSELoss expects float targets
    Y_pred, loss_dict = model.compute_loss(sample_labels, sample_input, sample_mask)
    assert "MSELoss" in loss_dict


def test_dsmil_rejects_multiclass():
    X = torch.randn(2, 10, 5)  # batch_size, bag_size, feat_dim
    mask = torch.ones(2, 10)

    model_binary = DSMIL(in_shape=(10, 5), n_outputs=1)
    assert model_binary.num_outputs == 1
    Y_pred = model_binary(X, mask)
    assert Y_pred.shape == (2,)

    with pytest.raises(ValueError, match="only supports binary classification"):
        DSMIL(in_shape=(10, 5), n_outputs=3)
