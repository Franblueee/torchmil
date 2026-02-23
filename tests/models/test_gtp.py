import torch
import torch.nn as nn
import pytest
from torchmil.models import GTP


@pytest.fixture
def sample_input():
    # Example input tensor: batch_size, bag_size, feature_dim
    return torch.randn(2, 10, 5)


@pytest.fixture
def sample_adj():
    # Example adjacency matrix: batch_size, bag_size, bag_size
    return torch.eye(10).unsqueeze(0).repeat(2, 1, 1)


@pytest.fixture
def sample_mask():
    # Example mask tensor: batch_size, bag_size
    return torch.randint(0, 2, (2, 10)).float()


@pytest.fixture
def sample_labels():
    # Example labels tensor: batch_size
    return torch.randint(0, 2, (2,)).float()


@pytest.fixture
def gtp_model():
    # Instantiate the GTP model with some default parameters
    return GTP(
        in_shape=(10, 5),
        att_dim=512,
        n_clusters=8,
        n_layers=2,
        n_heads=4,
        use_mlp=True,
        dropout=0.1,
    )


# Test the forward pass of the GTP model
def test_forward_pass(gtp_model, sample_input, sample_adj, sample_mask):
    Y_pred = gtp_model(sample_input, sample_adj, sample_mask)
    assert Y_pred.shape == (2,), "Output shape should be (batch_size,)"


def test_forward_pass_with_cam(gtp_model, sample_input, sample_adj, sample_mask):
    Y_pred, cam = gtp_model(sample_input, sample_adj, sample_mask, return_cam=True)
    assert Y_pred.shape == (2,), "Output shape should be (batch_size,)"
    assert cam.shape == (2, 10), "CAM shape should be (batch_size, bag_size)"


def test_forward_pass_with_loss(gtp_model, sample_input, sample_adj, sample_mask):
    Y_pred, loss_dict = gtp_model(
        sample_input, sample_adj, sample_mask, return_loss=True
    )
    assert Y_pred.shape == (2,), "Output shape should be (batch_size,)"
    assert isinstance(loss_dict, dict), "Loss should be a dictionary"
    assert "MinCutLoss" in loss_dict, "Loss dict should contain MinCut loss"
    assert "OrthoLoss" in loss_dict, "Loss dict should contain Ortho loss"


def test_compute_loss(gtp_model, sample_input, sample_adj, sample_mask, sample_labels):
    Y_pred, loss_dict = gtp_model.compute_loss(
        sample_labels, sample_input, sample_adj, sample_mask
    )
    assert Y_pred.shape == (2,), "Output shape should be (batch_size,)"
    assert isinstance(loss_dict, dict), "Loss should be a dictionary"
    assert "BCEWithLogitsLoss" in loss_dict, "Loss dict should contain BCE loss"
    assert "MinCutLoss" in loss_dict, "Loss dict should contain MinCut loss"
    assert "OrthoLoss" in loss_dict, "Loss dict should contain Ortho loss"


def test_predict(gtp_model, sample_input, sample_adj, sample_mask):
    Y_pred, inst_pred = gtp_model.predict(
        sample_input, sample_adj, sample_mask, return_inst_pred=True
    )
    assert Y_pred.shape == (2,), "Output shape should be (batch_size,)"
    assert inst_pred.shape == (
        2,
        10,
    ), "Instance predictions shape should be (batch_size, bag_size)"


# Test with a feature extractor
def test_with_feature_extractor(sample_input, sample_adj, sample_mask):
    feat_ext = nn.Sequential(nn.Linear(5, 16), nn.ReLU(), nn.Linear(16, 8))
    model = GTP(in_shape=(10, 5), feat_ext=feat_ext)
    Y_pred = model(sample_input, sample_adj, sample_mask)
    assert Y_pred.shape == (2,)


# Test with different number of clusters
def test_different_number_of_clusters(sample_input, sample_adj, sample_mask):
    model = GTP(in_shape=(10, 5), n_clusters=50)
    Y_pred = model(sample_input, sample_adj, sample_mask)
    assert Y_pred.shape == (2,)


def test_gtp_multiclass_support(sample_input, sample_adj, sample_mask):
    # Tests multiclass support with different numbers of outputs
    # Test binary classification (default)
    model_binary = GTP(in_shape=(10, 5), n_outputs=1)
    assert model_binary.num_outputs == 1
    Y_pred = model_binary(sample_input, sample_adj, sample_mask)
    assert Y_pred.shape == (2,)

    # Test multiclass with 3 classes
    model_3class = GTP(in_shape=(10, 5), n_outputs=3)
    assert model_3class.num_outputs == 3
    Y_pred = model_3class(sample_input, sample_adj, sample_mask)
    assert Y_pred.shape == (2, 3)

    # Test multiclass with 5 classes
    model_5class = GTP(in_shape=(10, 5), n_outputs=5)
    assert model_5class.num_outputs == 5
    Y_pred = model_5class(sample_input, sample_adj, sample_mask)
    assert Y_pred.shape == (2, 5)
