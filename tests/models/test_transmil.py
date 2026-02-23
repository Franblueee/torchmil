import pytest
import torch

from torchmil.models.transmil import (
    PPEG,
    TransMIL,
)  # Assuming your modules are in your_module.py


# PPEG Tests
class TestPPEG:
    @pytest.mark.parametrize("dim", [16, 32, 64])
    def test_ppeg_init(self, dim):
        ppeg = PPEG(dim)
        assert ppeg.proj.in_channels == dim
        assert ppeg.proj.out_channels == dim
        assert ppeg.proj1.in_channels == dim
        assert ppeg.proj1.out_channels == dim
        assert ppeg.proj2.in_channels == dim
        assert ppeg.proj2.out_channels == dim

    @pytest.mark.parametrize(
        "batch_size, H, W, dim", [(2, 8, 8, 16), (4, 16, 16, 32), (1, 32, 32, 64)]
    )
    def test_ppeg_forward(self, batch_size, H, W, dim):
        x = torch.randn(batch_size, H * W + 1, dim)
        ppeg = PPEG(dim)
        y = ppeg(x, H, W)
        assert y.shape == (batch_size, H * W + 1, dim)

    def test_ppeg_output_value(self):
        # Test that the output of PPEG is different from the input
        batch_size = 2
        H = 8
        W = 8
        dim = 16
        x = torch.randn(batch_size, H * W + 1, dim)
        ppeg = PPEG(dim)
        y = ppeg(x, H, W)
        assert not torch.allclose(x, y)


# TransMIL Tests
class TestTransMIL:
    @pytest.mark.parametrize(
        "in_shape, att_dim, n_layers, n_heads, n_landmarks, pinv_iterations, dropout, use_mlp",
        [
            ((10,), 32, 2, 4, 16, 6, 0.0, False),
            ((20,), 64, 3, 8, 32, 8, 0.1, True),
            ((10, 5), 32, 2, 4, 16, 6, 0.0, False),  # Multi-dimensional in_shape
        ],
    )
    def test_transmil_init(
        self,
        in_shape,
        att_dim,
        n_layers,
        n_heads,
        n_landmarks,
        pinv_iterations,
        dropout,
        use_mlp,
    ):
        model = TransMIL(
            in_shape=in_shape,
            att_dim=att_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            n_landmarks=n_landmarks,
            pinv_iterations=pinv_iterations,
            dropout=dropout,
            use_mlp=use_mlp,
        )
        assert model.layers[0].use_mlp == use_mlp
        assert len(model.layers) == n_layers

    @pytest.mark.parametrize(
        "batch_size, bag_size, in_dim", [(2, 10, 10), (4, 20, 20), (1, 5, 30)]
    )
    def test_transmil_forward(self, batch_size, bag_size, in_dim):
        in_shape = (in_dim,)  # Example in_shape
        att_dim = 32
        n_layers = 2
        n_heads = 4
        n_landmarks = 16
        pinv_iterations = 6
        dropout = 0.0
        use_mlp = False
        model = TransMIL(
            in_shape=in_shape,
            att_dim=att_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            n_landmarks=n_landmarks,
            pinv_iterations=pinv_iterations,
            dropout=dropout,
            use_mlp=use_mlp,
        )
        X = torch.randn(batch_size, bag_size, in_dim)
        Y_pred = model(X)
        assert Y_pred.shape == (batch_size,)

    def test_transmil_forward_return_att(self):
        batch_size = 2
        bag_size = 10
        in_dim = 10
        in_shape = (in_dim,)
        att_dim = 32
        n_layers = 2
        n_heads = 4
        n_landmarks = 16
        pinv_iterations = 6
        dropout = 0.0
        use_mlp = False
        model = TransMIL(
            in_shape=in_shape,
            att_dim=att_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            n_landmarks=n_landmarks,
            pinv_iterations=pinv_iterations,
            dropout=dropout,
            use_mlp=use_mlp,
        )
        X = torch.randn(batch_size, bag_size, in_dim)
        Y_pred, att = model(X, return_att=True)
        assert Y_pred.shape == (batch_size,)
        assert att.shape == (batch_size, bag_size)

    def test_transmil_compute_loss(self):
        batch_size = 2
        bag_size = 10
        in_dim = 10
        in_shape = (in_dim,)
        att_dim = 32
        n_layers = 2
        n_heads = 4
        n_landmarks = 16
        pinv_iterations = 6
        dropout = 0.0
        use_mlp = False
        model = TransMIL(
            in_shape=in_shape,
            att_dim=att_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            n_landmarks=n_landmarks,
            pinv_iterations=pinv_iterations,
            dropout=dropout,
            use_mlp=use_mlp,
        )
        X = torch.randn(batch_size, bag_size, in_dim)
        Y = torch.randint(0, 2, (batch_size,))  # Binary labels
        Y_pred, loss_dict = model.compute_loss(Y, X)
        assert Y_pred.shape == (batch_size,)
        assert "BCEWithLogitsLoss" in loss_dict  # Check for the default loss

    def test_transmil_predict(self):
        batch_size = 2
        bag_size = 10
        in_dim = 10
        in_shape = (in_dim,)
        att_dim = 32
        n_layers = 2
        n_heads = 4
        n_landmarks = 16
        pinv_iterations = 6
        dropout = 0.0
        use_mlp = False
        model = TransMIL(
            in_shape=in_shape,
            att_dim=att_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            n_landmarks=n_landmarks,
            pinv_iterations=pinv_iterations,
            dropout=dropout,
            use_mlp=use_mlp,
        )
        X = torch.randn(batch_size, bag_size, in_dim)
        Y_pred = model.predict(X, return_inst_pred=False)
        assert Y_pred.shape == (batch_size,)

    def test_transmil_predict_return_inst_pred(self):
        batch_size = 2
        bag_size = 10
        in_dim = 10
        in_shape = (in_dim,)
        att_dim = 32
        n_layers = 2
        n_heads = 4
        n_landmarks = 16
        pinv_iterations = 6
        dropout = 0.0
        use_mlp = False
        model = TransMIL(
            in_shape=in_shape,
            att_dim=att_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            n_landmarks=n_landmarks,
            pinv_iterations=pinv_iterations,
            dropout=dropout,
            use_mlp=use_mlp,
        )
        X = torch.randn(batch_size, bag_size, in_dim)
        Y_pred, att = model.predict(X, return_inst_pred=True)
        assert Y_pred.shape == (batch_size,)
        assert att.shape == (batch_size, bag_size)

    def test_transmil_multiclass_support(self):
        """Test multiclass support for TransMIL model."""
        batch_size = 2
        bag_size = 10
        in_dim = 10
        in_shape = (in_dim,)

        # Binary classification (n_outputs=1)
        model_binary = TransMIL(in_shape=in_shape, n_outputs=1)
        assert model_binary.num_outputs == 1
        X = torch.randn(batch_size, bag_size, in_dim)
        Y_pred = model_binary(X)
        assert Y_pred.shape == (batch_size,)

        # 3-class classification (n_outputs=3)
        model_3class = TransMIL(in_shape=in_shape, n_outputs=3)
        assert model_3class.num_outputs == 3
        Y_pred = model_3class(X)
        assert Y_pred.shape == (batch_size, 3)

        # 5-class classification (n_outputs=5)
        model_5class = TransMIL(in_shape=in_shape, n_outputs=5)
        assert model_5class.num_outputs == 5
        Y_pred = model_5class(X)
        assert Y_pred.shape == (batch_size, 5)
