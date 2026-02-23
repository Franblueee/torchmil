import torch
import pytest

# Import the modules to be tested
from torchmil.models import PatchGCN


# Fixtures for test data
@pytest.fixture
def sample_patch_gcn_data():
    batch_size = 2
    bag_size = 10
    input_shape = (256,)
    in_dim = 256
    # X: (batch_size, bag_size, *input_shape)
    X = torch.randn(batch_size, bag_size, *input_shape)
    # adj: (batch_size, bag_size, bag_size)
    adj = torch.randn(batch_size, bag_size, bag_size)
    # mask: (batch_size, bag_size)
    mask = torch.randint(0, 2, (batch_size, bag_size)).bool()
    # Y: (batch_size,)
    Y = torch.randint(0, 2, (batch_size,)).float()
    return X, adj, mask, Y, input_shape, in_dim


# Test PatchGCN
def test_patch_gcn(sample_patch_gcn_data):
    X, adj, mask, Y, input_shape, in_dim = sample_patch_gcn_data
    n_gcn_layers = 2
    mlp_depth = 1
    hidden_dim = 128
    att_dim = 64
    dropout = 0.1

    model = PatchGCN(
        in_shape=input_shape,
        n_gcn_layers=n_gcn_layers,
        mlp_depth=mlp_depth,
        hidden_dim=hidden_dim,
        att_dim=att_dim,
        dropout=dropout,
    )

    # Test forward pass
    Y_pred = model(X, adj, mask)
    assert Y_pred.shape == (X.shape[0],), "Output shape should be (batch_size,)"

    # Test compute_loss
    Y_pred, loss_dict = model.compute_loss(Y, X, adj, mask)
    assert Y_pred.shape == (X.shape[0],), "Output shape should be (batch_size,)"
    assert isinstance(loss_dict, dict), "Loss should be a dictionary"
    assert (
        "BCEWithLogitsLoss" in loss_dict
    ), "Loss dict should contain the criterion loss"

    # Test forward pass with attention
    Y_pred, att = model(X, adj, mask, return_att=True)
    assert Y_pred.shape == (X.shape[0],), "Output shape should be (batch_size,)"
    assert att.shape == (
        X.shape[0],
        X.shape[1],
    ), "Attention shape should be (batch_size, bag_size)"

    # Test predict
    Y_pred = model.predict(X, adj, mask, return_inst_pred=False)
    assert Y_pred.shape == (X.shape[0],), "Output shape should be (batch_size,)"

    Y_pred, y_inst_pred = model.predict(X, adj, mask, return_inst_pred=True)
    assert Y_pred.shape == (X.shape[0],), "Output shape should be (batch_size,)"
    assert y_inst_pred.shape == (
        X.shape[0],
        X.shape[1],
    ), "Instance prediction shape should be (batch_size, bag_size)"


def test_patch_gcn_multiclass_support(sample_patch_gcn_data):
    # Tests multiclass support with different numbers of outputs
    X, adj, mask, Y, input_shape, in_dim = sample_patch_gcn_data

    # Test binary classification (default)
    model_binary = PatchGCN(in_shape=input_shape, n_outputs=1)
    assert model_binary.num_outputs == 1
    Y_pred = model_binary(X, adj, mask)
    assert Y_pred.shape == (2,)

    # Test multiclass with 3 classes
    model_3class = PatchGCN(in_shape=input_shape, n_outputs=3)
    assert model_3class.num_outputs == 3
    Y_pred = model_3class(X, adj, mask)
    assert Y_pred.shape == (2, 3)

    # Test multiclass with 5 classes
    model_5class = PatchGCN(in_shape=input_shape, n_outputs=5)
    assert model_5class.num_outputs == 5
    Y_pred = model_5class(X, adj, mask)
    assert Y_pred.shape == (2, 5)
