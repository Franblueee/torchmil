import pytest
import torch
from torch.nn.attention import SDPBackend

from torchmil.nn.attention import (
    MultiheadSelfAttention,
)  # Assuming your script is saved as your_module.py

SDP_BACKEND = [
    SDPBackend.MATH,
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION,
    SDPBackend.CUDNN_ATTENTION,
]


class TestMultiheadSelfAttention:
    @pytest.mark.parametrize(
        "in_dim, out_dim, att_dim, n_heads, dropout, learn_weights",
        [
            (10, 20, 32, 2, 0.1, True),
            (20, 15, 64, 4, 0.0, False),
            (8, None, 128, 8, 0.2, True),
            (10, 5, 32, 2, 0.1, True),  # out_dim < att_dim
            (30, 20, 32, 2, 0.1, False),
        ],
    )  # learn_weights = False
    def test_forward_pass(
        self, in_dim, out_dim, att_dim, n_heads, dropout, learn_weights
    ):
        sample_input = torch.randn(2, 5, in_dim)  # batch_size, seq_len, in_dim

        if att_dim % n_heads != 0:
            with pytest.raises(AssertionError):
                MultiheadSelfAttention(
                    in_dim, out_dim, att_dim, n_heads, dropout, learn_weights
                )
            return

        layer = MultiheadSelfAttention(
            in_dim, out_dim, att_dim, n_heads, dropout, learn_weights
        )
        output = layer(sample_input)
        expected_out_dim = out_dim if out_dim is not None else in_dim
        assert output.shape == (2, 5, expected_out_dim)

    def test_forward_pass_with_mask(self):
        in_dim = 10
        att_dim = 32
        n_heads = 2
        sample_input = torch.randn(2, 5, in_dim)  # batch_size, seq_len, in_dim
        sample_mask = torch.randint(0, 2, (2, 5)).bool()  # batch_size, seq_len

        if att_dim % n_heads != 0:
            pytest.skip("att_dim must be divisible by n_heads")
        layer = MultiheadSelfAttention(in_dim, att_dim=att_dim, n_heads=n_heads)
        output = layer(sample_input, mask=sample_mask)
        assert output.shape == (2, 5, in_dim)

    def test_forward_pass_return_att(self):
        in_dim = 10
        att_dim = 32
        n_heads = 2
        sample_input = torch.randn(2, 5, in_dim)  # batch_size, seq_len, in_dim

        if att_dim % n_heads != 0:
            pytest.skip("att_dim must be divisible by n_heads")
        layer = MultiheadSelfAttention(in_dim, att_dim=att_dim, n_heads=n_heads)
        output, attention = layer(sample_input, return_att=True)
        assert output.shape == (2, 5, in_dim)
        assert attention.shape == (2, n_heads, 5, 5)

    def test_invalid_n_heads(self):
        with pytest.raises(AssertionError):
            MultiheadSelfAttention(
                in_dim=10, att_dim=32, n_heads=3
            )  # att_dim not divisible by n_heads

    def test_no_weights(self):
        in_dim = 10
        out_dim = 20
        att_dim = 32
        n_heads = 2
        sample_input = torch.randn(2, 5, in_dim)  # batch_size, seq_len, in_dim

        if att_dim % n_heads != 0:
            pytest.skip("att_dim must be divisible by n_heads")
        layer = MultiheadSelfAttention(
            in_dim, out_dim, att_dim, n_heads, learn_weights=False
        )
        output = layer(sample_input)
        assert output.shape == (2, 5, out_dim)

    @pytest.mark.parametrize("training", [False, True])
    @pytest.mark.parametrize("return_att", [False, True])
    def test_dropout_respects_training_mode(self, training, return_att):
        # Full dropout makes the expected output deterministic in both modes.
        layer = MultiheadSelfAttention(
            in_dim=4, n_heads=4, dropout=1.0, learn_weights=False
        ).train(training)
        sample_input = torch.ones(1, 8, 4)

        output = layer(sample_input, return_att=return_att)
        if return_att:
            output, _ = output

        expected = torch.zeros_like(sample_input) if training else sample_input
        torch.testing.assert_close(output, expected)
