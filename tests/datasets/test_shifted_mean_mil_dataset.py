import pytest
import torch
from tensordict import TensorDict

from torchmil.datasets.shifted_mean_mil_dataset import (
    ShiftedMeanMILDataset,
    normal_pdf,
)


@pytest.fixture
def base_dataset():
    return ShiftedMeanMILDataset(
        N=8,
        R=2,
        S_low=5,
        S_high=7,
        K=2,
        M=6,
        p_y1=0.5,
        Delta=1.5,
        seed=123,
    )


def test_normal_pdf_properties():
    x = torch.tensor([0.0, 1.0, -1.0])
    pdf = normal_pdf(x)
    expected_at_zero = 1.0 / torch.sqrt(torch.tensor(2.0 * torch.pi))

    assert torch.isclose(pdf[0], expected_at_zero)
    assert torch.isclose(pdf[1], pdf[2])
    assert torch.all(pdf > 0)


def test_shifted_mean_dataset_init_and_len(base_dataset):
    ds = base_dataset
    assert ds.N == 8
    assert ds.R == 2
    assert ds.K == 2
    assert ds.M == 6
    assert len(ds) == 8
    assert len(ds.lengths) == 8
    assert len(ds.H_split) == 8
    assert len(ds.y_inst_list) == 8


def test_shifted_mean_dataset_getitem_structure(base_dataset):
    bag = base_dataset[0]

    assert isinstance(bag, TensorDict)
    assert set(bag.keys()) == {"X", "Y", "y_inst", "bag_size"}

    X = bag["X"]
    Y = bag["Y"]
    y_inst = bag["y_inst"]
    bag_size = bag["bag_size"]

    assert X.ndim == 2
    assert X.shape[1] == base_dataset.M
    assert y_inst.ndim == 1
    assert X.shape[0] == y_inst.shape[0]
    assert X.shape[0] == bag_size.item()
    assert Y.dtype == torch.int64
    assert y_inst.dtype == torch.int64
    assert bag_size.dtype == torch.int64


def test_shifted_mean_dataset_getitem_bounds(base_dataset):
    with pytest.raises(IndexError, match="out of range"):
        _ = base_dataset[len(base_dataset)]


def test_positive_bags_have_contiguous_positive_instance_block():
    ds = ShiftedMeanMILDataset(
        N=6,
        R=3,
        S_low=6,
        S_high=8,
        K=1,
        M=4,
        p_y1=1.0,
        Delta=1.0,
        seed=7,
    )

    for i in range(len(ds)):
        bag = ds[i]
        assert bag["Y"].item() == 1

        y_inst = bag["y_inst"]
        pos_idx = (y_inst == 1).nonzero(as_tuple=True)[0]

        assert pos_idx.numel() == ds.R
        assert torch.all(pos_idx[1:] - pos_idx[:-1] == 1)

        start = ds.u[i].item()
        expected = torch.zeros(ds.lengths[i], dtype=torch.long)
        expected[start : start + ds.R] = 1
        assert torch.equal(y_inst, expected)


def test_negative_bags_have_no_positive_instances():
    ds = ShiftedMeanMILDataset(
        N=6,
        R=2,
        S_low=5,
        S_high=7,
        K=2,
        M=5,
        p_y1=0.0,
        Delta=2.0,
        seed=11,
    )

    for i in range(len(ds)):
        bag = ds[i]
        assert bag["Y"].item() == 0
        assert torch.all(bag["y_inst"] == 0)


def test_shifted_mean_dataset_determinism_with_seed():
    kwargs = dict(
        N=5,
        R=2,
        S_low=4,
        S_high=6,
        K=2,
        M=4,
        p_y1=0.4,
        Delta=1.2,
        seed=99,
    )
    ds1 = ShiftedMeanMILDataset(**kwargs)
    ds2 = ShiftedMeanMILDataset(**kwargs)

    assert ds1.lengths == ds2.lengths
    assert torch.equal(ds1.H, ds2.H)
    assert torch.equal(ds1.u, ds2.u)
    assert torch.equal(ds1.y, ds2.y)

    for i in range(len(ds1)):
        b1 = ds1[i]
        b2 = ds2[i]
        assert torch.equal(b1["X"], b2["X"])
        assert b1["Y"].item() == b2["Y"].item()
        assert torch.equal(b1["y_inst"], b2["y_inst"])
        assert b1["bag_size"].item() == b2["bag_size"].item()


def test_posterior_probability_matches_extreme_priors():
    ds_all_pos = ShiftedMeanMILDataset(
        N=3,
        R=1,
        S_low=3,
        S_high=3,
        K=1,
        M=2,
        p_y1=1.0,
        seed=5,
    )
    ds_all_neg = ShiftedMeanMILDataset(
        N=3,
        R=1,
        S_low=3,
        S_high=3,
        K=1,
        M=2,
        p_y1=0.0,
        seed=5,
    )

    for i in range(3):
        p_pos = ds_all_pos.p_y1_given_h(i)
        p_neg = ds_all_neg.p_y1_given_h(i)

        assert torch.isfinite(p_pos)
        assert torch.isfinite(p_neg)
        assert torch.isclose(p_pos, torch.tensor(1.0), atol=1e-6)
        assert torch.isclose(p_neg, torch.tensor(0.0), atol=1e-6)
