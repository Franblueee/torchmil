import pytest
import torch

from torchmil.models.sm_abmil import SmABMIL
from torchmil.nn import SmAttentionPool


@pytest.fixture
def sample_inputs():
    """Provides sample input tensors for testing."""
    batch_size = 2
    bag_size = 5
    feat_dim = 10
    in_shape = (bag_size, feat_dim)  # Example in_shape for a single instance

    X = torch.randn(batch_size, bag_size, feat_dim)
    adj = torch.randint(0, 2, (batch_size, bag_size, bag_size)).float()
    mask = torch.randint(0, 2, (batch_size, bag_size)).bool()
    Y = torch.randint(0, 2, (batch_size,)).float()  # Binary labels

    return X, adj, mask, Y, in_shape, batch_size, bag_size, feat_dim


def test_smabmil_init_default(sample_inputs):
    """Test model initialization with default parameters."""
    _, _, _, _, in_shape, _, _, _ = sample_inputs
    model = SmABMIL(in_shape=in_shape)

    assert isinstance(model, SmABMIL)
    assert isinstance(model.feat_ext, torch.nn.Identity)
    assert isinstance(model.pool, SmAttentionPool)
    assert model.pool.att_dim == 128
    assert model.pool.act == "tanh"
    assert model.pool.sm_mode == "approx"
    assert model.pool.sm_alpha == "trainable"
    assert model.pool.sm_steps == 10
    assert model.pool.sm_where == "early"
    assert not model.pool.spectral_norm
    assert isinstance(model.last_layer, torch.nn.Linear)
    assert model.last_layer.in_features == in_shape[-1]
    assert model.last_layer.out_features == 1
    assert isinstance(model.criterion, torch.nn.BCEWithLogitsLoss)


@pytest.mark.parametrize(
    "att_dim, att_act, sm_mode, sm_alpha, sm_steps, sm_where, spectral_norm",
    [
        (64, "relu", "exact", 0.5, 5, "mid", True),
        (256, "gelu", "approx", "trainable", 20, "late", False),
    ],
)
def test_smabmil_init_custom(
    sample_inputs,
    att_dim,
    att_act,
    sm_mode,
    sm_alpha,
    sm_steps,
    sm_where,
    spectral_norm,
):
    """Test model initialization with custom parameters."""
    _, _, _, _, in_shape, _, _, _ = sample_inputs
    model = SmABMIL(
        in_shape=in_shape,
        att_dim=att_dim,
        att_act=att_act,
        sm_mode=sm_mode,
        sm_alpha=sm_alpha,
        sm_steps=sm_steps,
        sm_where=sm_where,
        spectral_norm=spectral_norm,
        feat_ext=torch.nn.Linear(in_shape[-1], in_shape[-1]),  # Custom feat_ext
        criterion=torch.nn.MSELoss(),  # Custom criterion
    )

    assert isinstance(model, SmABMIL)
    assert isinstance(model.feat_ext, torch.nn.Linear)
    assert isinstance(model.pool, SmAttentionPool)
    assert model.pool.att_dim == att_dim
    assert model.pool.act == att_act
    assert model.pool.sm_mode == sm_mode
    assert model.pool.sm_alpha == sm_alpha
    assert model.pool.sm_steps == sm_steps
    assert model.pool.sm_where == sm_where
    assert model.pool.spectral_norm == spectral_norm
    assert isinstance(model.last_layer, torch.nn.Linear)
    assert model.last_layer.in_features == in_shape[-1]
    assert model.last_layer.out_features == 1
    assert isinstance(model.criterion, torch.nn.MSELoss)


def test_smabmil_forward_no_att(sample_inputs):
    """Test forward pass without returning attention."""
    X, adj, mask, _, in_shape, batch_size, _, _ = sample_inputs
    model = SmABMIL(in_shape=in_shape)
    Y_pred = model.forward(X, adj, mask, return_att=False)

    assert isinstance(Y_pred, torch.Tensor)
    assert Y_pred.shape == (batch_size,)  # (batch_size,) for logits


def test_smabmil_forward_with_att(sample_inputs):
    """Test forward pass with returning attention."""
    X, adj, mask, _, in_shape, batch_size, bag_size, _ = sample_inputs
    model = SmABMIL(in_shape=in_shape)
    Y_pred, att = model.forward(X, adj, mask, return_att=True)

    assert isinstance(Y_pred, torch.Tensor)
    assert Y_pred.shape == (batch_size,)  # (batch_size,) for logits
    assert isinstance(att, torch.Tensor)
    assert att.shape == (batch_size, bag_size)  # (batch_size, bag_size) for attention


def test_smabmil_compute_loss(sample_inputs):
    """Test compute_loss method."""
    X, adj, mask, Y, in_shape, batch_size, _, _ = sample_inputs
    model = SmABMIL(in_shape=in_shape)
    Y_pred, loss_dict = model.compute_loss(Y, X, adj, mask)

    assert isinstance(Y_pred, torch.Tensor)
    assert Y_pred.shape == (batch_size,)
    assert isinstance(loss_dict, dict)
    assert len(loss_dict) == 1
    assert "BCEWithLogitsLoss" in loss_dict
    assert isinstance(loss_dict["BCEWithLogitsLoss"], torch.Tensor)
    assert loss_dict["BCEWithLogitsLoss"].ndim == 0  # Scalar loss


def test_smabmil_predict_bag_only(sample_inputs):
    """Test predict method returning only bag predictions."""
    X, adj, mask, _, in_shape, batch_size, _, _ = sample_inputs
    model = SmABMIL(in_shape=in_shape)
    Y_pred = model.predict(X, adj, mask, return_inst_pred=False)

    assert isinstance(Y_pred, torch.Tensor)
    assert Y_pred.shape == (batch_size,)


def test_smabmil_predict_with_inst_pred(sample_inputs):
    """Test predict method returning bag and instance predictions."""
    X, adj, mask, _, in_shape, batch_size, bag_size, _ = sample_inputs
    model = SmABMIL(in_shape=in_shape)
    Y_pred, y_inst_pred = model.predict(X, adj, mask, return_inst_pred=True)

    assert isinstance(Y_pred, torch.Tensor)
    assert Y_pred.shape == (batch_size,)
    assert isinstance(y_inst_pred, torch.Tensor)
    assert y_inst_pred.shape == (batch_size, bag_size)


def test_smabmil_multiclass_support(sample_inputs):
    """Test multiclass support for SmABMIL model."""
    X, adj, mask, _, in_shape, batch_size, _, _ = sample_inputs

    # Binary classification (n_outputs=1)
    model_binary = SmABMIL(in_shape=in_shape, n_outputs=1)
    assert model_binary.num_outputs == 1
    Y_pred = model_binary.forward(X, adj, mask)
    assert Y_pred.shape == (batch_size,)

    # 3-class classification (n_outputs=3)
    model_3class = SmABMIL(in_shape=in_shape, n_outputs=3)
    assert model_3class.num_outputs == 3
    Y_pred = model_3class.forward(X, adj, mask)
    assert Y_pred.shape == (batch_size, 3)

    # 5-class classification (n_outputs=5)
    model_5class = SmABMIL(in_shape=in_shape, n_outputs=5)
    assert model_5class.num_outputs == 5
    Y_pred = model_5class.forward(X, adj, mask)
    assert Y_pred.shape == (batch_size, 5)
