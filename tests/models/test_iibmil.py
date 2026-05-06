import torch
import pytest

# Import the modules to be tested
from torchmil.models.iibmil import IIBMIL


# Fixtures for test data
@pytest.fixture
def sample_input_iibmil():
    batch_size = 2
    n_instances = 10
    n_queries = 5
    in_dim = 256
    # X: (batch_size, n_instances, in_dim)
    X = torch.randn(batch_size, n_instances, in_dim)
    # U: (batch_size, n_queries, in_dim)
    U = torch.randn(batch_size, n_queries, in_dim)
    # mask: (batch_size, n_instances)
    mask = torch.randint(0, 2, (batch_size, n_instances)).bool()
    return X, U, mask


@pytest.fixture
def sample_iibmil_data():
    batch_size = 2
    bag_size = 10
    input_shape = (256,)
    # X: (batch_size, bag_size, *input_shape)
    X = torch.randn(batch_size, bag_size, *input_shape)
    # Y: (batch_size,)
    Y = torch.randint(0, 2, (batch_size,)).float()
    # mask: (batch_size, bag_size)
    mask = torch.randint(0, 2, (batch_size, bag_size)).bool()
    return X, Y, mask, input_shape


# Test IIBMIL
def test_iibmil(sample_iibmil_data):
    X, Y, mask, input_shape = sample_iibmil_data
    att_dim = 128
    n_layers_encoder = 1
    n_layers_decoder = 1
    n_heads = 4

    model = IIBMIL(
        in_shape=input_shape,
        att_dim=att_dim,
        n_layers_encoder=n_layers_encoder,
        n_layers_decoder=n_layers_decoder,
        n_heads=n_heads,
    )

    # Test forward pass
    Y_pred = model(X, mask)
    assert Y_pred.shape == (X.shape[0],), "Output shape should be (batch_size,)"

    # Test compute_loss
    Y_pred, loss_dict = model.compute_loss(Y, X, mask)
    assert Y_pred.shape == (X.shape[0],), "Output shape should be (batch_size,)"
    assert isinstance(loss_dict, dict), "Loss should be a dictionary"
    assert (
        "BCEWithLogitsLoss" in loss_dict
    ), "Loss dict should contain the criterion loss"
    assert "InstLoss" in loss_dict, "Loss dict should contain the instance loss"

    # Test predict
    Y_pred, y_inst_pred = model.predict(X, mask, return_inst_pred=True)
    assert Y_pred.shape == (X.shape[0],), "Output shape should be (batch_size,)"
    assert y_inst_pred.shape == (
        X.shape[0],
        X.shape[1],
    ), "Instance prediction shape should be (batch_size, bag_size)"

    Y_pred = model.predict(X, mask, return_inst_pred=False)
    assert Y_pred.shape == (X.shape[0],), "Output shape should be (batch_size,)"


def test_iibmil_rejects_multiclass(sample_iibmil_data):
    X, Y, mask, input_shape = sample_iibmil_data

    model_binary = IIBMIL(in_shape=input_shape, n_outputs=1)
    assert model_binary.num_outputs == 1
    Y_pred = model_binary(X, mask)
    assert Y_pred.shape == (2,)

    with pytest.raises(ValueError, match="only supports binary classification"):
        IIBMIL(in_shape=input_shape, n_outputs=3)
