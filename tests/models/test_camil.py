import pytest
import torch

from torchmil.data import collate_fn
from torchmil.models import CAMIL, MILModelWrapper
from torchmil.models.camil import CAMILSelfAttention


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
def camil_model():
    return CAMIL(
        in_shape=(256,),
        nystrom_att_dim=256,
        pool_att_dim=64,
        n_heads=2,
        n_landmarks=4,
        feat_ext=torch.nn.Identity(),
        gated_pool=True,
    )


def test_forward_output_shape(camil_model, dummy_inputs):
    X, adj, mask, _ = dummy_inputs
    Y_pred = camil_model(X, adj, mask)
    assert Y_pred.shape == (X.shape[0],), "Output shape mismatch for CAMIL forward pass"


def test_forward_with_attention_output(camil_model, dummy_inputs):
    X, adj, mask, _ = dummy_inputs
    Y_pred, att = camil_model(X, adj, mask, return_att=True)
    assert Y_pred.shape == (X.shape[0],)
    assert att.shape == (X.shape[0], X.shape[1]), "Attention output shape mismatch"


def test_compute_loss(camil_model, dummy_inputs):
    X, adj, mask, Y = dummy_inputs
    Y_pred, loss_dict = camil_model.compute_loss(Y, X, adj, mask)
    assert isinstance(loss_dict, dict), "Loss output should be a dictionary"
    for name, loss in loss_dict.items():
        assert (
            torch.is_tensor(loss) and loss.requires_grad
        ), f"Loss {name} is not a valid differentiable tensor"


def test_predict_with_instance_output(camil_model, dummy_inputs):
    X, adj, mask, _ = dummy_inputs
    Y_pred, inst_pred = camil_model.predict(X, adj, mask, return_inst_pred=True)
    assert Y_pred.shape == (X.shape[0],)
    assert inst_pred.shape == (
        X.shape[0],
        X.shape[1],
    ), "Instance prediction shape mismatch"


def test_self_attention_ignores_masked_padding():
    # The weights are normalized over the instances of the bag, so padded
    # instances must not take a share of them.
    torch.manual_seed(0)
    layer = CAMILSelfAttention(in_dim=8, att_dim=8)
    X = torch.randn(1, 3, 8)
    adj = torch.rand(1, 3, 3)

    # The padding holds arbitrary values, which the mask must hide.
    padded_X = torch.full((1, 5, 8), 7.0)
    padded_X[:, :3] = X
    padded_adj = torch.full((1, 5, 5), 7.0)
    padded_adj[:, :3, :3] = adj
    padded_mask = torch.tensor([[1, 1, 1, 0, 0]])

    with torch.no_grad():
        expected = layer(X, adj)
        actual = layer(padded_X, padded_adj, padded_mask)

    torch.testing.assert_close(actual[:, :3], expected)
    assert (actual[:, 3:] == 0).all()


@pytest.mark.parametrize("padding_first", [False, True])
@pytest.mark.parametrize("use_mlp, gated_pool", [(False, False), (True, True)])
def test_masked_padding_does_not_change_prediction(padding_first, use_mlp, gated_pool):
    torch.manual_seed(0)
    model = CAMIL(
        in_shape=(8,),
        nystrom_att_dim=16,
        n_heads=2,
        n_landmarks=4,
        use_mlp=use_mlp,
        gated_pool=gated_pool,
    )
    model.eval()
    small_X = torch.randn(1, 4, 8)
    small_adj = torch.rand(1, 4, 4)
    large_X = torch.randn(1, 9, 8)
    large_adj = torch.rand(1, 9, 9)

    # The padding holds arbitrary values, which the mask must hide.
    small_mask = torch.zeros(9, dtype=torch.bool)
    if padding_first:
        small_mask[5:] = True
    else:
        small_mask[:4] = True
    padded_X = torch.full((2, 9, 8), 7.0)
    padded_X[0, small_mask] = small_X[0]
    padded_X[1] = large_X[0]
    padded_adj = torch.full((2, 9, 9), 7.0)
    padded_adj[0][small_mask[:, None] & small_mask[None, :]] = small_adj.flatten()
    padded_adj[1] = large_adj[0]
    padded_mask = torch.stack([small_mask, torch.ones(9, dtype=torch.bool)])

    with torch.no_grad():
        small_pred, small_att = model(small_X, small_adj, return_att=True)
        large_pred, large_att = model(large_X, large_adj, return_att=True)
        expected = torch.cat([small_pred, large_pred])
        actual, actual_att = model(padded_X, padded_adj, padded_mask, return_att=True)
        loss_pred, _ = model.compute_loss(
            torch.ones(2), padded_X, padded_adj, padded_mask
        )
        predict_pred = model.predict(
            padded_X, padded_adj, padded_mask, return_inst_pred=False
        )

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(loss_pred, expected)
    torch.testing.assert_close(predict_pred, expected)
    torch.testing.assert_close(actual_att[0, small_mask], small_att[0])
    torch.testing.assert_close(actual_att[1], large_att[0])


def test_full_mask_matches_no_mask(camil_model, dummy_inputs):
    X, adj, mask, _ = dummy_inputs
    camil_model.eval()

    with torch.no_grad():
        expected = camil_model(X, adj)
        actual = camil_model(X, adj, mask)

    torch.testing.assert_close(actual, expected)


# The path used in practice: bags collated into a batch and passed through the wrapper
def test_collated_batch_matches_single_bags():
    torch.manual_seed(0)
    # More landmarks than instances in the smallest bag
    model = MILModelWrapper(
        CAMIL(in_shape=(8,), nystrom_att_dim=16, n_heads=2, n_landmarks=8)
    )
    model.eval()
    bags = [
        {
            "X": torch.randn(n, 8),
            "adj": torch.rand(n, n).to_sparse(),
            "Y": torch.tensor(1.0),
        }
        for n in (2, 13, 30, 5)
    ]

    with torch.no_grad():
        expected = torch.cat([model(collate_fn([bag])) for bag in bags])
        actual = model(collate_fn(bags))

    torch.testing.assert_close(actual, expected)
