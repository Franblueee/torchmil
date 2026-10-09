import numpy as np
import torch
import torch.nn as nn
import pytest

from torchmil.models import (
    DTFDMIL,
)  # Import the DTFDMIL class.  Assuming it is in your_module.py


@pytest.fixture
def sample_input():
    # Example input tensor: batch_size, bag_size, feature_dim
    return torch.randn(2, 20, 10)


@pytest.fixture
def sample_mask():
    # Example mask tensor:  batch_size, bag_size
    return torch.randint(0, 2, (2, 20))


@pytest.fixture
def sample_labels():
    # Example labels tensor: batch_size
    return torch.randint(0, 2, (2,)).float()


@pytest.fixture
def dtf_dsmil_model():
    # Instantiate the DTFDMIL model with some default parameters
    return DTFDMIL(in_shape=(20, 10), att_dim=128, n_groups=4, distill_mode="maxmin")


# Test the forward pass of the DTFDMIL model
def test_forward_pass(dtf_dsmil_model, sample_input, sample_mask):
    Y_pred = dtf_dsmil_model(sample_input, sample_mask)
    assert Y_pred.shape == (2,), "Output shape should be (batch_size,)"


def test_forward_pass_with_pseudo_predictions(
    dtf_dsmil_model, sample_input, sample_mask
):
    Y_pred, pseudo_pred = dtf_dsmil_model(
        sample_input, sample_mask, return_pseudo_pred=True
    )
    assert Y_pred.shape == (2,), "Output shape should be (batch_size,)"
    assert (
        pseudo_pred.shape[0] == 2
    ), "Pseudo predictions shape should be (batch_size, n_groups)"
    assert pseudo_pred.shape[1] == dtf_dsmil_model.n_groups


def test_forward_pass_with_instance_cam(dtf_dsmil_model, sample_input, sample_mask):
    Y_pred, inst_cam = dtf_dsmil_model(sample_input, sample_mask, return_inst_cam=True)
    assert Y_pred.shape == (2,), "Output shape should be (batch_size,)"
    assert inst_cam.shape == (
        2,
        20,
    ), "Instance CAM shape should be (batch_size, bag_size)"


def test_compute_loss(dtf_dsmil_model, sample_input, sample_mask, sample_labels):
    Y_pred, loss_dict = dtf_dsmil_model.compute_loss(
        sample_labels, sample_input, sample_mask
    )
    assert Y_pred.shape == (2,), "Output shape should be (batch_size,)"
    assert isinstance(loss_dict, dict), "Loss should be a dictionary"
    assert "BCEWithLogitsLoss_t1" in loss_dict, "Loss dict should contain tier 1 loss"
    assert "BCEWithLogitsLoss_t2" in loss_dict, "Loss dict should contain tier 2 loss"


def test_predict(dtf_dsmil_model, sample_input, sample_mask):
    Y_pred, inst_pred = dtf_dsmil_model.predict(
        sample_input, sample_mask, return_inst_pred=True
    )
    assert Y_pred.shape == (2,), "Output shape should be (batch_size,)"
    assert inst_pred.shape == (
        2,
        20,
    ), "Instance predictions shape should be (batch_size, bag_size)"


# Test with different distillation modes
@pytest.mark.parametrize("distill_mode", ["maxmin", "max", "afs"])
def test_different_distill_modes(distill_mode, sample_input, sample_mask):
    model = DTFDMIL(in_shape=(20, 10), distill_mode=distill_mode)
    Y_pred = model(sample_input, sample_mask)
    assert Y_pred.shape == (2,)


# Test with a different number of groups
def test_different_number_of_groups(sample_input, sample_mask):
    model = DTFDMIL(
        in_shape=(20, 10), n_groups=8
    )  # Test with a different number of groups
    Y_pred = model(sample_input, sample_mask)
    assert Y_pred.shape == (2,)


# Test with a feature extractor
def test_with_feature_extractor(sample_input, sample_mask):
    feat_ext = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 16))
    model = DTFDMIL(in_shape=(20, 10), feat_ext=feat_ext)
    Y_pred = model(sample_input, sample_mask)
    assert Y_pred.shape == (2,)


def test_with_invalid_distill_mode(sample_input, sample_mask):
    with pytest.raises(ValueError):
        DTFDMIL(in_shape=(20, 10), distill_mode="invalid_mode")


# Masked padding must not change the bag prediction
@pytest.mark.parametrize("distill_mode", ["maxmin", "max", "afs"])
def test_masked_padding_does_not_change_prediction(distill_mode):
    torch.manual_seed(0)
    model = DTFDMIL(in_shape=(3,), att_dim=2, n_groups=1, distill_mode=distill_mode)
    model.eval()
    X = torch.randn(2, 5, 3)
    mask = torch.ones(2, 5, dtype=torch.bool)
    padded_X = torch.nn.functional.pad(X, (0, 0, 0, 4))
    padded_mask = torch.nn.functional.pad(mask, (0, 4), value=False)

    # One group keeps every real instance together regardless of the shuffle.
    with torch.no_grad():
        np.random.seed(0)
        expected = model(X, mask)
        np.random.seed(0)
        actual = model(padded_X, padded_mask)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("distill_mode", ["maxmin", "max", "afs"])
def test_forward_pass_without_mask(distill_mode, sample_input):
    model = DTFDMIL(in_shape=(20, 10), distill_mode=distill_mode)
    Y_pred = model(sample_input)
    assert Y_pred.shape == (2,)
