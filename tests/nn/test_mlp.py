import torch
import pytest

from torchmil.nn.mlp import MLP, get_activation


# Test activation function utility
def test_get_activation():
    # Test different activation functions
    relu = get_activation("relu")
    assert isinstance(relu, torch.nn.ReLU)

    sigmoid = get_activation("sigmoid")
    assert isinstance(sigmoid, torch.nn.Sigmoid)

    tanh = get_activation("tanh")
    assert isinstance(tanh, torch.nn.Tanh)

    # Test None case
    none_activation = get_activation("none")
    assert none_activation is None


# Fixtures for common setup
@pytest.fixture
def sample_input():
    return torch.randn(5, 10)  # batch_size=5, input_dim=10


@pytest.fixture
def mlp_basic():
    return MLP(input_size=10, linear_sizes=[5, 3], activations=["relu", "relu"])


# Basic tests for MLP class
def test_mlp_initialization():
    # Test basic initialization
    mlp = MLP(input_size=10, linear_sizes=[5, 3], activations=["relu", "relu"])
    assert mlp.input_size == 10
    assert mlp.linear_sizes == [5, 3]
    assert mlp.activations == ["relu", "relu"]


def test_mlp_initialization_default_params():
    # Test initialization with default parameters
    mlp = MLP()
    assert mlp.input_size == 512
    assert mlp.linear_sizes == [100, 50]
    assert mlp.activations == ["relu", "relu"]


def test_mlp_forward_pass(sample_input, mlp_basic):
    # Test basic forward pass
    output = mlp_basic(sample_input)
    assert output.shape == (5, 3)  # batch_size, final_layer_size


def test_mlp_no_activation():
    # Test with no activation (None activation)
    mlp = MLP(input_size=10, linear_sizes=[5], activations=["None"])
    input_data = torch.randn(2, 10)
    output = mlp(input_data)
    assert output.shape == (2, 5)


def test_mlp_different_activations():
    # Test different activation functions work
    input_data = torch.randn(3, 10)

    # ReLU activation
    mlp_relu = MLP(input_size=10, linear_sizes=[5], activations=["relu"])
    output_relu = mlp_relu(input_data)
    assert output_relu.shape == (3, 5)

    # Sigmoid activation
    mlp_sigmoid = MLP(input_size=10, linear_sizes=[5], activations=["sigmoid"])
    output_sigmoid = mlp_sigmoid(input_data)
    assert output_sigmoid.shape == (3, 5)


def test_mlp_multiple_layers():
    # Test MLP with multiple layers
    mlp = MLP(
        input_size=20, linear_sizes=[16, 8, 4], activations=["relu", "relu", "None"]
    )
    input_data = torch.randn(2, 20)
    output = mlp(input_data)
    assert output.shape == (2, 4)
