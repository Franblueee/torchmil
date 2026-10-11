import pytest
import torch

from torchmil.nn.attention.nystrom_attention import (
    NystromAttention,
    moore_penrose_iter_pinv,
)


def create_random_input(batch_size, seq_len, in_dim):
    return torch.randn(batch_size, seq_len, in_dim)


def create_random_mask(batch_size, seq_len):
    return torch.randint(0, 2, (batch_size, seq_len)).bool()


class TestNystromAttention:
    @pytest.mark.parametrize(
        "in_dim, out_dim, att_dim, n_heads, n_landmarks, learn_weights, return_att",
        [
            (32, 64, 128, 4, 32, True, False),
            (16, 16, 64, 2, 64, False, True),
            (24, 32, 96, 3, 128, True, False),
            (32, 64, 128, 4, 20, True, False),  # n_landmarks < seq_len
            (32, 64, 128, 4, 256, True, True),
        ],
    )
    def test_forward_pass(
        self, in_dim, out_dim, att_dim, n_heads, n_landmarks, learn_weights, return_att
    ):
        batch_size = 2
        seq_len = 100
        x = create_random_input(batch_size, seq_len, in_dim)
        layer = NystromAttention(
            in_dim, out_dim, att_dim, n_heads, learn_weights, n_landmarks
        )
        output = layer(x, return_att=return_att)

        expected_out_dim = out_dim if out_dim is not None else in_dim
        if return_att:
            y, att = output
            assert y.shape == (batch_size, seq_len, expected_out_dim)
            assert att.shape == (batch_size, n_heads, seq_len, seq_len)
        else:
            y = output
            assert y.shape == (batch_size, seq_len, expected_out_dim)

    @pytest.mark.parametrize(
        "in_dim, att_dim, n_heads, n_landmarks", [(32, 128, 4, 32)]
    )
    def test_forward_pass_with_mask(self, in_dim, att_dim, n_heads, n_landmarks):
        batch_size = 2
        seq_len = 100
        x = create_random_input(batch_size, seq_len, in_dim)
        mask = create_random_mask(batch_size, seq_len)
        layer = NystromAttention(
            in_dim, att_dim=att_dim, n_heads=n_heads, n_landmarks=n_landmarks
        )
        output = layer(x, mask=mask)
        assert output.shape == (batch_size, seq_len, in_dim)

    @pytest.mark.parametrize(
        "in_dim, att_dim, n_heads, n_landmarks", [(32, 128, 4, 32)]
    )
    def test_forward_pass_variable_seq_len(self, in_dim, att_dim, n_heads, n_landmarks):
        batch_size = 2
        seq_len1 = 100
        seq_len2 = 120
        x1 = create_random_input(batch_size, seq_len1, in_dim)
        x2 = create_random_input(batch_size, seq_len2, in_dim)
        layer = NystromAttention(
            in_dim, att_dim=att_dim, n_heads=n_heads, n_landmarks=n_landmarks
        )
        output1 = layer(x1)
        output2 = layer(x2)
        assert output1.shape == (batch_size, seq_len1, in_dim)
        assert output2.shape == (batch_size, seq_len2, in_dim)

    def test_no_weights(self):
        in_dim = 32
        out_dim = 64
        att_dim = 128
        n_heads = 4
        n_landmarks = 32
        batch_size = 2
        seq_len = 100
        x = create_random_input(batch_size, seq_len, in_dim)
        layer = NystromAttention(
            in_dim,
            out_dim,
            att_dim,
            n_heads,
            learn_weights=False,
            n_landmarks=n_landmarks,
        )
        output = layer(x)
        assert output.shape == (batch_size, seq_len, out_dim)

    # A sequence must get the same output alone and padded in a batch with a longer one.
    # Its landmarks come from its own length (5), not from the padded length (9).
    @pytest.mark.parametrize("learn_weights", [True, False])
    def test_masked_padding_does_not_change_output(self, learn_weights):
        torch.manual_seed(0)
        layer = NystromAttention(
            in_dim=8, att_dim=8, n_heads=2, n_landmarks=4, learn_weights=learn_weights
        )
        short_x = torch.randn(1, 5, 8)
        long_x = torch.randn(1, 9, 8)
        # The padding holds arbitrary values, which the mask must hide.
        padded_x = torch.cat(
            [torch.nn.functional.pad(short_x, (0, 0, 0, 4), value=7.0), long_x]
        )
        mask = torch.ones(2, 9, dtype=torch.bool)
        mask[0, 5:] = False

        with torch.no_grad():
            short_out, short_att = layer(short_x, return_att=True)
            long_out, long_att = layer(long_x, return_att=True)
            actual_out, actual_att = layer(padded_x, mask=mask, return_att=True)

        torch.testing.assert_close(actual_out[:1, :5], short_out)
        torch.testing.assert_close(actual_out[1:], long_out)
        torch.testing.assert_close(actual_att[:1, :, :5, :5], short_att)
        torch.testing.assert_close(actual_att[1:], long_att)
        # masked positions produce nothing and receive no attention
        assert (actual_out[0, 5:] == 0).all()
        assert (actual_att[0, :, 5:, :] == 0).all()
        assert (actual_att[0, :, :, 5:] == 0).all()

    def test_masked_positions_can_be_anywhere(self):
        torch.manual_seed(0)
        layer = NystromAttention(in_dim=8, att_dim=8, n_heads=2, n_landmarks=4)
        x = torch.randn(1, 5, 8)
        mask = torch.tensor([[False, True, True, False, True, False, True, True]])
        scattered_x = torch.full((1, 8, 8), 7.0)
        scattered_x[mask] = x[0]

        with torch.no_grad():
            expected = layer(x)
            actual = layer(scattered_x, mask=mask)

        torch.testing.assert_close(actual[mask], expected[0])
        assert (actual[~mask] == 0).all()

    def test_full_mask_matches_no_mask(self):
        torch.manual_seed(0)
        layer = NystromAttention(in_dim=8, att_dim=8, n_heads=2, n_landmarks=4)
        x = torch.randn(2, 9, 8)

        with torch.no_grad():
            expected = layer(x)
            actual = layer(x, mask=torch.ones(2, 9))

        torch.testing.assert_close(actual, expected)

    def test_sequences_in_a_batch_do_not_affect_each_other(self):
        torch.manual_seed(0)
        layer = NystromAttention(in_dim=8, att_dim=8, n_heads=2, n_landmarks=4)
        x = torch.randn(3, 9, 8)

        with torch.no_grad():
            expected = torch.cat([layer(x[i : i + 1]) for i in range(3)])
            actual = layer(x)

        torch.testing.assert_close(actual, expected)


def test_pinv_batch_elements_do_not_affect_each_other():
    # The iteration starts from a scaled matrix. Scaling with the largest values of
    # the whole batch would make each result depend on the other matrices.
    torch.manual_seed(0)
    x = torch.randn(3, 2, 4, 4).softmax(dim=-1)
    x[1] = x[1] * 100

    expected = torch.cat([moore_penrose_iter_pinv(x[i : i + 1]) for i in range(3)])
    actual = moore_penrose_iter_pinv(x)

    torch.testing.assert_close(actual, expected)


def test_pinv_single_matrix():
    torch.manual_seed(0)
    x = torch.randn(4, 4).softmax(dim=-1)

    expected = moore_penrose_iter_pinv(x[None])
    actual = moore_penrose_iter_pinv(x)

    torch.testing.assert_close(actual, expected)
