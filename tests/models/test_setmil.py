import torch
import pytest

from torchmil.models.setmil import SETMIL, PMF


# Fixtures for test data
@pytest.fixture
def sample_setmil_data():
    batch_size = 2
    bag_size = 10
    in_shape = (256,)
    feat_dim = 256
    coord_dim = 2
    # X: (batch_size, bag_size, *in_shape)
    X = torch.randn(batch_size, bag_size, *in_shape)
    # coords: (batch_size, bag_size, 2)
    coords = torch.randint(0, 64, (batch_size, bag_size, coord_dim)).float()
    # Y: (batch_size,)
    Y = torch.randint(0, 2, (batch_size,)).float()
    return X, coords, Y, in_shape, feat_dim


# Test PMF module
def test_pmf():
    batch_size = 2
    in_dim = 3
    coord1 = 64
    coord2 = 64
    out_dim = 512
    # X: (batch_size, in_dim, coord1, coord2)
    X = torch.randn(batch_size, in_dim, coord1, coord2)
    pmf = PMF(in_dim=in_dim, out_dim=out_dim)
    Y = pmf(X)
    assert Y.shape == (batch_size, 4096, out_dim), "Output shape of PMF is incorrect"


# Test SETMIL model
def test_setmil(sample_setmil_data):
    X, coords, Y, in_shape, feat_dim = sample_setmil_data
    att_dim = 512
    set_n_layers = 2
    set_n_heads = 4

    model = SETMIL(
        in_shape=in_shape,
        att_dim=att_dim,
        set_n_layers=set_n_layers,
        set_n_heads=set_n_heads,
    )

    # Test forward pass
    Y_pred = model(X, coords)
    assert Y_pred.shape == (X.shape[0],), "Output shape should be (batch_size,)"

    # Test compute_loss
    Y_pred, loss_dict = model.compute_loss(Y, X, coords)
    assert Y_pred.shape == (X.shape[0],), "Output shape should be (batch_size,)"
    assert isinstance(loss_dict, dict), "Loss should be a dictionary"
    assert (
        "BCEWithLogitsLoss" in loss_dict
    ), "Loss dict should contain the criterion loss"

    # Test forward pass with attention
    Y_pred, att = model(X, coords, return_att=True)
    assert Y_pred.shape == (X.shape[0],), "Output shape should be (batch_size,)"
    assert att.shape == (
        X.shape[0],
        X.shape[1],
    ), "Attention shape should be (batch_size, bag_size)"

    # Test predict
    Y_pred = model.predict(X, coords, return_inst_pred=False)
    assert Y_pred.shape == (X.shape[0],), "Output shape should be (batch_size,)"

    Y_pred, y_inst_pred = model.predict(X, coords, return_inst_pred=True)
    assert Y_pred.shape == (X.shape[0],), "Output shape should be (batch_size,)"
    assert y_inst_pred.shape == (
        X.shape[0],
        X.shape[1],
    ), "Instance prediction shape should be (batch_size, bag_size)"


def test_setmil_multiclass_support(sample_setmil_data):
    """Test multiclass support for SETMIL model."""
    X, coords, Y, in_shape, feat_dim = sample_setmil_data
    att_dim = 512
    set_n_layers = 2
    set_n_heads = 4

    # Binary classification (n_outputs=1)
    model_binary = SETMIL(
        in_shape=in_shape,
        att_dim=att_dim,
        set_n_layers=set_n_layers,
        set_n_heads=set_n_heads,
        n_outputs=1,
    )
    assert model_binary.num_outputs == 1
    Y_pred = model_binary(X, coords)
    assert Y_pred.shape == (X.shape[0],)

    # 3-class classification (n_outputs=3)
    model_3class = SETMIL(
        in_shape=in_shape,
        att_dim=att_dim,
        set_n_layers=set_n_layers,
        set_n_heads=set_n_heads,
        n_outputs=3,
    )
    assert model_3class.num_outputs == 3
    Y_pred = model_3class(X, coords)
    assert Y_pred.shape == (X.shape[0], 3)

    # 5-class classification (n_outputs=5)
    model_5class = SETMIL(
        in_shape=in_shape,
        att_dim=att_dim,
        set_n_layers=set_n_layers,
        set_n_heads=set_n_heads,
        n_outputs=5,
    )
    assert model_5class.num_outputs == 5
    Y_pred = model_5class(X, coords)
    assert Y_pred.shape == (X.shape[0], 5)
