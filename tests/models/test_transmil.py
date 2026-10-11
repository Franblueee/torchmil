import math

import pytest
import torch

from torchmil.data import collate_fn
from torchmil.models import MILModelWrapper
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

    def test_ppeg_small_grid_inside_larger_grid(self):
        # A 2x2 grid placed in the top-left corner of a 3x3 grid of zeros gets the
        # same values as on its own: the zeros act as the border of the convolutions.
        torch.manual_seed(0)
        dim = 4
        ppeg = PPEG(dim)
        cls_token = torch.randn(1, 1, dim)
        small_grid = torch.randn(1, 2, 2, dim)
        large_grid = torch.zeros(1, 3, 3, dim)
        large_grid[:, :2, :2] = small_grid

        with torch.no_grad():
            expected = ppeg(
                torch.cat([cls_token, small_grid.flatten(1, 2)], dim=1), 2, 2
            )
            actual = ppeg(torch.cat([cls_token, large_grid.flatten(1, 2)], dim=1), 3, 3)

        expected_grid = expected[:, 1:].view(1, 2, 2, dim)
        actual_grid = actual[:, 1:].view(1, 3, 3, dim)
        torch.testing.assert_close(actual_grid[:, :2, :2], expected_grid)
        torch.testing.assert_close(actual[:, 0], expected[:, 0])


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

    def test_transmil_sequence_squaring(self):
        # Each bag gets the smallest square grid that fits its own instances,
        # completed by repeating its first instances. Smaller grids sit in the
        # top-left corner of the largest one, and the rest of the cells are empty.
        model = TransMIL(in_shape=(1,), att_dim=1, n_heads=1)
        X = torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0],  # five instances: 3x3 grid
                [6.0, 7.0, 8.0, 0.0, 0.0],  # three instances and padding: 2x2 grid
            ]
        ).unsqueeze(-1)
        mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]]).bool()

        grid, grid_mask, side, grid_size = model._sequence_squaring(X, mask)

        assert grid_size == 3
        assert side.tolist() == [3, 2]
        assert grid.view(2, 3, 3).tolist() == [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 1.0], [2.0, 3.0, 4.0]],
            [[6.0, 7.0, 0.0], [8.0, 6.0, 0.0], [0.0, 0.0, 0.0]],
        ]
        assert grid_mask.view(2, 3, 3).tolist() == [
            [[True, True, True], [True, True, True], [True, True, True]],
            [[True, True, False], [True, True, False], [False, False, False]],
        ]

    # A bag must get the same prediction alone and padded in a batch with a larger bag
    @pytest.mark.parametrize("n_layers, use_mlp", [(2, False), (3, True)])
    @pytest.mark.parametrize("padding_first", [False, True])
    def test_transmil_masked_padding_does_not_change_prediction(
        self, n_layers, use_mlp, padding_first
    ):
        torch.manual_seed(0)
        model = TransMIL(
            in_shape=(8,),
            att_dim=16,
            n_layers=n_layers,
            n_heads=2,
            n_landmarks=4,
            use_mlp=use_mlp,
        )
        model.eval()
        small_X = torch.randn(1, 4, 8)
        large_X = torch.randn(1, 9, 8)

        # The padding holds arbitrary values, which the mask must hide.
        small_mask = torch.zeros(9, dtype=torch.bool)
        if padding_first:
            small_mask[5:] = True
        else:
            small_mask[:4] = True
        padded_X = torch.full((2, 9, 8), 7.0)
        padded_X[0, small_mask] = small_X[0]
        padded_X[1] = large_X[0]
        padded_mask = torch.stack([small_mask, torch.ones(9, dtype=torch.bool)])

        with torch.no_grad():
            small_pred, small_att = model(small_X, return_att=True)
            large_pred, large_att = model(large_X, return_att=True)
            expected = torch.cat([small_pred, large_pred])
            actual, actual_att = model(padded_X, padded_mask, return_att=True)
            loss_pred, _ = model.compute_loss(torch.ones(2), padded_X, padded_mask)
            predict_pred = model.predict(padded_X, padded_mask, return_inst_pred=False)

        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(loss_pred, expected)
        torch.testing.assert_close(predict_pred, expected)
        torch.testing.assert_close(actual_att[0, small_mask], small_att[0])
        torch.testing.assert_close(actual_att[1], large_att[0])
        assert (actual_att[0, ~small_mask] == 0).all()

    def test_transmil_full_mask_matches_no_mask(self):
        torch.manual_seed(0)
        model = TransMIL(in_shape=(8,), att_dim=16, n_heads=2, n_landmarks=4)
        model.eval()
        X = torch.randn(2, 9, 8)

        with torch.no_grad():
            expected = model(X)
            actual = model(X, torch.ones(2, 9))

        torch.testing.assert_close(actual, expected)

    def test_transmil_bags_in_a_batch_do_not_affect_each_other(self):
        torch.manual_seed(0)
        model = TransMIL(in_shape=(8,), att_dim=16, n_heads=2, n_landmarks=4)
        model.eval()
        X = torch.randn(3, 9, 8)

        with torch.no_grad():
            expected = torch.cat([model(X[i : i + 1]) for i in range(3)])
            actual = model(X)

        torch.testing.assert_close(actual, expected)

    def test_transmil_bag_without_instances(self):
        # Sequence squaring has nothing to build a grid from.
        model = TransMIL(in_shape=(8,), att_dim=16, n_heads=2, n_landmarks=4)
        X = torch.randn(2, 4, 8)
        mask = torch.tensor([[1, 1, 1, 1], [0, 0, 0, 0]]).bool()

        with pytest.raises(ValueError, match="at least one instance"):
            model(X, mask)

    def test_transmil_grid_side_is_smallest_that_fits(self):
        # One bag of each size from 1 to 300 instances.
        model = TransMIL(in_shape=(1,), att_dim=1, n_heads=1)
        sizes = torch.arange(1, 301)
        mask = torch.arange(300)[None, :] < sizes[:, None]
        X = torch.zeros(300, 300, 1)

        _, _, side, _ = model._sequence_squaring(X, mask)

        expected = [math.isqrt(n - 1) + 1 for n in sizes.tolist()]
        assert side.tolist() == expected

    # The path used in practice: bags collated into a batch and passed through the wrapper
    def test_transmil_collated_batch_matches_single_bags(self):
        torch.manual_seed(0)
        # More landmarks than tokens in the smallest bag
        model = MILModelWrapper(
            TransMIL(in_shape=(8,), att_dim=16, n_heads=2, n_landmarks=8)
        )
        model.eval()
        bags = [
            {"X": torch.randn(n, 8), "Y": torch.tensor(1.0)} for n in (2, 13, 30, 5)
        ]

        with torch.no_grad():
            expected = torch.cat([model(collate_fn([bag])) for bag in bags])
            actual = model(collate_fn(bags))

        torch.testing.assert_close(actual, expected)
