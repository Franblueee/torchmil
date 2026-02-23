import pytest
import torch

from torchmil.models.sm_transformer_abmil import SmTransformerABMIL
from torchmil.nn.transformers.sm_transformer import SmTransformerEncoder
from torchmil.nn.attention.sm_attention_pool import SmAttentionPool


@pytest.fixture
def sample_inputs_sm_transformer():
    """Provides sample input tensors for SmTransformerABMIL testing."""
    batch_size = 2
    bag_size = 4
    in_dim = 256  # Feature dimension
    in_shape = (in_dim,)  # Input shape for a single instance

    X = torch.randn(batch_size, bag_size, *in_shape)
    adj = torch.randn(batch_size, bag_size, bag_size)
    mask = torch.randint(0, 2, (batch_size, bag_size)).bool()
    Y = torch.randint(0, 2, (batch_size,)).float()  # Binary labels

    return X, adj, mask, Y, in_shape, batch_size, bag_size, in_dim


def test_sm_transformer_abmil_initialization_default(sample_inputs_sm_transformer):
    """Test SmTransformerABMIL initialization with default parameters."""
    _, _, _, _, in_shape, _, _, in_dim = sample_inputs_sm_transformer
    model = SmTransformerABMIL(in_shape=in_shape)

    assert isinstance(model, SmTransformerABMIL)
    assert isinstance(model.feat_ext, torch.nn.Identity)
    assert isinstance(model.transformer_encoder, SmTransformerEncoder)
    assert isinstance(model.pool, SmAttentionPool)
    assert isinstance(model.last_layer, torch.nn.Linear)
    assert isinstance(model.criterion, torch.nn.BCEWithLogitsLoss)


def test_sm_transformer_abmil_initialization_custom(sample_inputs_sm_transformer):
    """Test SmTransformerABMIL initialization with custom parameters."""
    _, _, _, _, in_shape, _, _, in_dim = sample_inputs_sm_transformer

    class CustomFeatExt(torch.nn.Module):
        def __init__(self, in_dim_arg):
            super().__init__()
            self.linear = torch.nn.Linear(in_dim_arg, 128)

        def forward(self, x):
            return self.linear(x)

    feat_ext = CustomFeatExt(in_dim)
    pool_att_dim = 64
    pool_act = "relu"
    pool_sm_mode = "exact"
    pool_sm_alpha = 0.8
    pool_sm_steps = 5
    pool_sm_where = "late"
    pool_spectral_norm = True
    transf_att_dim = 256
    transf_n_layers = 2
    transf_n_heads = 8
    transf_use_mlp = False
    transf_add_self = False
    transf_dropout = 0.2
    transf_sm_alpha = 0.2
    transf_sm_mode = "approx"
    transf_sm_steps = 8
    criterion = torch.nn.CrossEntropyLoss()

    model = SmTransformerABMIL(
        in_shape=in_shape,
        pool_att_dim=pool_att_dim,
        pool_act=pool_act,
        pool_sm_mode=pool_sm_mode,
        pool_sm_alpha=pool_sm_alpha,
        pool_sm_steps=pool_sm_steps,
        pool_sm_where=pool_sm_where,
        pool_spectral_norm=pool_spectral_norm,
        feat_ext=feat_ext,
        transf_att_dim=transf_att_dim,
        transf_n_layers=transf_n_layers,
        transf_n_heads=transf_n_heads,
        transf_use_mlp=transf_use_mlp,
        transf_add_self=transf_add_self,
        transf_dropout=transf_dropout,
        transf_sm_alpha=transf_sm_alpha,
        transf_sm_mode=transf_sm_mode,
        transf_sm_steps=transf_sm_steps,
        criterion=criterion,
    )

    assert isinstance(model, SmTransformerABMIL)
    assert isinstance(model.feat_ext, CustomFeatExt)
    assert isinstance(model.transformer_encoder, SmTransformerEncoder)

    assert isinstance(model.pool, SmAttentionPool)

    assert isinstance(model.last_layer, torch.nn.Linear)
    assert model.last_layer.in_features == feat_ext.linear.out_features
    assert model.last_layer.out_features == 1
    assert isinstance(model.criterion, torch.nn.CrossEntropyLoss)


def test_sm_transformer_abmil_forward(sample_inputs_sm_transformer):
    """Test forward pass of SmTransformerABMIL."""
    X, adj, mask, _, in_shape, batch_size, bag_size, _ = sample_inputs_sm_transformer
    model = SmTransformerABMIL(in_shape=in_shape)

    # Test forward pass without attention
    Y_pred = model(X, adj, mask, return_att=False)
    assert isinstance(Y_pred, torch.Tensor)
    assert Y_pred.shape == (batch_size,)

    # Test forward pass with attention
    Y_pred, att = model(X, adj, mask, return_att=True)
    assert isinstance(Y_pred, torch.Tensor)
    assert Y_pred.shape == (batch_size,)
    assert isinstance(att, torch.Tensor)
    assert att.shape == (batch_size, bag_size)  # Assertion for attention shape


def test_sm_transformer_abmil_compute_loss(sample_inputs_sm_transformer):
    """Test compute_loss method of SmTransformerABMIL."""
    X, adj, mask, Y, in_shape, batch_size, _, _ = sample_inputs_sm_transformer
    model = SmTransformerABMIL(in_shape=in_shape)

    Y_pred, loss_dict = model.compute_loss(Y, X, adj, mask)
    assert isinstance(Y_pred, torch.Tensor)
    assert Y_pred.shape == (batch_size,)
    assert isinstance(loss_dict, dict)
    assert len(loss_dict) == 1
    assert "BCEWithLogitsLoss" in loss_dict
    assert isinstance(loss_dict["BCEWithLogitsLoss"], torch.Tensor)
    assert loss_dict["BCEWithLogitsLoss"].ndim == 0  # Scalar loss


def test_sm_transformer_abmil_predict(sample_inputs_sm_transformer):
    """Test predict method of SmTransformerABMIL."""
    X, adj, mask, _, in_shape, batch_size, bag_size, _ = sample_inputs_sm_transformer
    model = SmTransformerABMIL(in_shape=in_shape)

    # Test predict without instance predictions
    Y_pred = model.predict(X, adj, mask, return_inst_pred=False)
    assert isinstance(Y_pred, torch.Tensor)
    assert Y_pred.shape == (batch_size,)

    # Test predict with instance predictions
    Y_pred, y_inst_pred = model.predict(X, adj, mask, return_inst_pred=True)
    assert isinstance(Y_pred, torch.Tensor)
    assert Y_pred.shape == (batch_size,)
    assert isinstance(y_inst_pred, torch.Tensor)
    assert y_inst_pred.shape == (batch_size, bag_size)


def test_sm_transformer_abmil_multiclass_support(sample_inputs_sm_transformer):
    """Test multiclass support for SmTransformerABMIL model."""
    X, adj, mask, _, in_shape, batch_size, _, _ = sample_inputs_sm_transformer

    # Binary classification (n_outputs=1)
    model_binary = SmTransformerABMIL(in_shape=in_shape, n_outputs=1)
    assert model_binary.num_outputs == 1
    Y_pred = model_binary(X, adj, mask)
    assert Y_pred.shape == (batch_size,)

    # 3-class classification (n_outputs=3)
    model_3class = SmTransformerABMIL(in_shape=in_shape, n_outputs=3)
    assert model_3class.num_outputs == 3
    Y_pred = model_3class(X, adj, mask)
    assert Y_pred.shape == (batch_size, 3)

    # 5-class classification (n_outputs=5)
    model_5class = SmTransformerABMIL(in_shape=in_shape, n_outputs=5)
    assert model_5class.num_outputs == 5
    Y_pred = model_5class(X, adj, mask)
    assert Y_pred.shape == (batch_size, 5)
