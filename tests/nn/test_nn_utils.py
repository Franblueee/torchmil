import torch
import pytest

from torchmil.nn.utils import (
    LazyLinear,
    masked_softmax,
    MaskedSoftmax,
    get_feat_dim,
    SinusoidalPositionalEncodingND,
    log_sum_exp,
    delta,
    SmoothTop1SVM,
)


# LazyLinear tests
def test_lazy_linear_lazy_init():
    linear = LazyLinear(out_features=10)
    x = torch.randn(5, 20)
    result = linear(x)
    assert result.shape == (5, 10)


def test_lazy_linear_eager_init():
    linear = LazyLinear(in_features=20, out_features=10)
    x = torch.randn(5, 20)
    result = linear(x)
    assert result.shape == (5, 10)


# masked_softmax tests
def test_masked_softmax_no_mask():
    x = torch.randn(2, 3, 4)
    result = masked_softmax(x)
    assert result.shape == x.shape


def test_masked_softmax_with_mask():
    x = torch.randn(2, 3, 4)
    mask = torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.bool)
    result = masked_softmax(x, mask)
    assert result.shape == x.shape


def test_masked_softmax_mask_dim_mismatch():
    x = torch.randn(2, 3, 4)
    mask = torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.bool)
    result = masked_softmax(x, mask)
    assert result.shape == x.shape


# MaskedSoftmax module tests
def test_masked_softmax_module():
    x = torch.randn(2, 3)
    mask = torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.bool)
    masked_softmax_module = MaskedSoftmax()
    result = masked_softmax_module(x, mask)
    assert result.shape == x.shape


# get_feat_dim tests
def test_get_feat_dim():
    linear = torch.nn.Linear(10, 20)
    feat_dim = get_feat_dim(linear, (10,))
    assert feat_dim == 20


# SinusoidalPositionalEncodingND tests
def test_sinusoidal_positional_encoding_nd_2d():
    encoding = SinusoidalPositionalEncodingND(n_dim=2, channels=16)
    tensor = torch.randn(2, 4, 4, 16)
    result = encoding(tensor)
    assert result.shape == tensor.shape


def test_sinusoidal_positional_encoding_nd_3d():
    encoding = SinusoidalPositionalEncodingND(n_dim=3, channels=16)
    tensor = torch.randn(2, 4, 4, 4, 16)
    result = encoding(tensor)
    assert result.shape == tensor.shape


def test_sinusoidal_positional_encoding_nd_invalid_shape():
    encoding = SinusoidalPositionalEncodingND(n_dim=2, channels=16)
    tensor = torch.randn(2, 4, 16)
    with pytest.raises(RuntimeError):
        encoding(tensor)


# log_sum_exp tests
def test_log_sum_exp():
    x = torch.randn(3, 5)
    result = log_sum_exp(x)
    assert result.shape == (3,)


# delta tests
def test_delta():
    y = torch.tensor([0, 1, 2])
    labels = torch.tensor([0, 1, 2, 3])
    result = delta(y, labels)
    expected = torch.tensor([[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1]]).float()
    torch.testing.assert_close(result, expected)


# SmoothTop1SVM tests
@pytest.mark.parametrize("column_targets", [False, True])
def test_smooth_top1_svm(column_targets):
    x = torch.tensor([[2.0, 0.0]], requires_grad=True)
    y = torch.tensor([0])
    if column_targets:
        y = y[:, None].float()  # CLAM's target shape and dtype.

    loss = SmoothTop1SVM(n_classes=2)(x, y)
    torch.testing.assert_close(loss, torch.nn.functional.softplus(torch.tensor(-1.0)))
    loss.backward()
    expected_grad = torch.tensor([[-1.0, 1.0]]) * torch.sigmoid(torch.tensor(-1.0))
    torch.testing.assert_close(x.grad, expected_grad)


@pytest.mark.parametrize("column_targets", [False, True])
def test_smooth_top1_svm_batch_duplication(column_targets):
    x = torch.zeros(1, 2)
    y = torch.tensor([0])
    if column_targets:
        y = y[:, None]
    loss_fn = SmoothTop1SVM(n_classes=2)
    expected = torch.nn.functional.softplus(torch.tensor(1.0))

    torch.testing.assert_close(loss_fn(x, y), expected)
    duplicated = loss_fn(x.repeat(2, 1), y.repeat_interleave(2, dim=0))
    torch.testing.assert_close(duplicated, expected)


@pytest.mark.parametrize("column_targets", [False, True])
@pytest.mark.parametrize("batch_size", [1, 3])
def test_smooth_top1_svm_hard_loss(column_targets, batch_size):
    x = torch.tensor([[0.0, 10.0], [10.0, 0.0], [0.0, 10.0]])[:batch_size]
    y = torch.tensor([0, 0, 1])[:batch_size]
    if column_targets:
        y = y[:, None]
    loss_fn = SmoothTop1SVM(n_classes=2)
    expected = torch.tensor([11.0, 0.0, 0.0])[:batch_size]

    torch.testing.assert_close(loss_fn.hard_loss(x, y), expected)
    torch.testing.assert_close(loss_fn(x, y), expected.mean())
