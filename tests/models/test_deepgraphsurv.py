import pytest
import torch
from torchmil.models import deepgraphsurv


@pytest.fixture
def dummy_inputs():
    batch_size = 2
    bag_size = 5
    feat_dim = 256

    X = torch.randn(batch_size, bag_size, feat_dim)
    adj = torch.rand(batch_size, bag_size, bag_size)
    mask = torch.ones(batch_size, bag_size)

    Y = torch.randint(0, 2, (batch_size,)).float()
    return X, adj, mask, Y


@pytest.fixture
def deepgraphsurv_model():
    return deepgraphsurv.DeepGraphSurv(
        in_shape=(256,),
        hidden_dim=128,
        att_dim=64,
        n_layers_rep=2,
        n_layers_att=2,
        dropout=0.1,
        feat_ext=torch.nn.Identity(),
    )


def test_forward_output_shape(deepgraphsurv_model, dummy_inputs):
    X, adj, mask, _ = dummy_inputs
    Y_pred = deepgraphsurv_model(X, adj, mask)
    assert Y_pred.shape == (
        X.shape[0],
    ), "Output shape mismatch for DeepGraphSurv forward pass"


def test_forward_with_attention_output(deepgraphsurv_model, dummy_inputs):
    X, adj, mask, _ = dummy_inputs
    Y_pred, att = deepgraphsurv_model(X, adj, mask, return_att=True)
    assert Y_pred.shape == (X.shape[0],)
    assert att.shape == (X.shape[0], X.shape[1]), "Attention output shape mismatch"


def test_compute_loss(deepgraphsurv_model, dummy_inputs):
    X, adj, mask, Y = dummy_inputs
    Y_pred, loss_dict = deepgraphsurv_model.compute_loss(Y, X, adj, mask)
    assert isinstance(loss_dict, dict), "Loss output should be a dictionary"
    for name, loss in loss_dict.items():
        assert (
            torch.is_tensor(loss) and loss.requires_grad
        ), f"Loss {name} is not a valid differentiable tensor"


def test_predict_with_instance_output(deepgraphsurv_model, dummy_inputs):
    X, adj, mask, _ = dummy_inputs
    Y_pred, inst_pred = deepgraphsurv_model.predict(X, adj, mask, return_inst_pred=True)
    assert Y_pred.shape == (X.shape[0],)
    assert inst_pred.shape == (
        X.shape[0],
        X.shape[1],
    ), "Instance prediction shape mismatch"


def test_deepgraphsurv_multiclass_support(dummy_inputs):
    # Tests multiclass support with different numbers of outputs
    X, adj, mask, _ = dummy_inputs

    # Test binary classification (default)
    model_binary = deepgraphsurv.DeepGraphSurv(in_shape=(256,), n_outputs=1)
    assert model_binary.num_outputs == 1
    Y_pred = model_binary(X, adj, mask)
    assert Y_pred.shape == (2,)

    # Test multiclass with 3 classes
    model_3class = deepgraphsurv.DeepGraphSurv(in_shape=(256,), n_outputs=3)
    assert model_3class.num_outputs == 3
    Y_pred = model_3class(X, adj, mask)
    assert Y_pred.shape == (2, 3)

    # Test multiclass with 5 classes
    model_5class = deepgraphsurv.DeepGraphSurv(in_shape=(256,), n_outputs=5)
    assert model_5class.num_outputs == 5
    Y_pred = model_5class(X, adj, mask)
    assert Y_pred.shape == (2, 5)
