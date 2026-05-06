import torch
import pytest

from torchmil.models import CLAM_SB


@pytest.fixture
def batch_size():
    return 2


@pytest.fixture
def bag_size():
    return 5


@pytest.fixture
def in_shape():
    return (10,)


@pytest.fixture
def features(batch_size, bag_size, in_shape):
    return torch.randn(batch_size, bag_size, *in_shape)


@pytest.fixture
def mask(batch_size, bag_size):
    return torch.randint(0, 2, (batch_size, bag_size), dtype=torch.bool)


@pytest.fixture
def labels(batch_size):
    return torch.randint(0, 2, (batch_size,))


def test_clam_sb_forward(features, mask):
    model = CLAM_SB(in_shape=(10,))
    Y_pred = model(features, mask)
    assert Y_pred.shape == (features.shape[0],)


def test_clam_sb_forward_return_att(features, mask):
    model = CLAM_SB(in_shape=(10,))
    Y_pred, att = model(features, mask, return_att=True)
    assert Y_pred.shape == (features.shape[0],)
    assert att.shape == (features.shape[0], features.shape[1])


def test_clam_sb_forward_return_emb(features, mask):
    model = CLAM_SB(in_shape=(10,))
    Y_pred, emb = model(features, mask, return_emb=True)
    assert Y_pred.shape == (features.shape[0],)
    assert emb.shape == features.shape


def test_clam_sb_compute_loss(features, mask, labels):
    model = CLAM_SB(in_shape=(10,))
    Y_pred, loss_dict = model.compute_loss(labels, features, mask)
    assert Y_pred.shape == (features.shape[0],)
    assert "BCEWithLogitsLoss" in loss_dict
    assert "InstLoss" in loss_dict


def test_clam_sb_predict(features, mask):
    model = CLAM_SB(in_shape=(10,))
    Y_pred, y_inst_pred = model.predict(features, mask)
    assert Y_pred.shape == (features.shape[0],)
    assert y_inst_pred.shape == (features.shape[0], features.shape[1])


def test_clam_sb_predict_return_inst_pred_false(features, mask):
    model = CLAM_SB(in_shape=(10,))
    Y_pred = model.predict(features, mask, return_inst_pred=False)
    assert Y_pred.shape == (features.shape[0],)


def test_clam_sb_inst_eval(features):
    model = CLAM_SB(in_shape=(10,))
    att = torch.randn(features.shape[1])
    emb = torch.randn(features.shape[1], 10)
    instance_loss, all_preds, all_targets = model.inst_eval(
        att, emb, model.inst_classifiers[0]
    )
    assert instance_loss.shape == ()
    assert all_preds.shape == (2 * min(model.k_sample, features.shape[1]), 1)
    assert all_targets.shape == (2 * min(model.k_sample, features.shape[1]),)


def test_clam_sb_inst_eval_out(features):
    model = CLAM_SB(in_shape=(10,))
    att = torch.randn(features.shape[1])
    emb = torch.randn(features.shape[1], 10)
    instance_loss, p_preds, p_targets = model.inst_eval_out(
        att, emb, model.inst_classifiers[0]
    )
    assert instance_loss.shape == ()
    assert p_preds.shape == (min(model.k_sample, features.shape[1]), 1)
    assert p_targets.shape == (min(model.k_sample, features.shape[1]),)


def test_clam_sb_compute_inst_loss(features, labels):
    model = CLAM_SB(in_shape=(10,))
    att = torch.randn(features.shape[0], features.shape[1])
    emb = torch.randn(features.shape[0], features.shape[1], 10)
    inst_loss = model.compute_inst_loss(att, emb, labels)
    assert inst_loss.shape == ()


def test_clam_sb_invalid_inst_loss_name():
    with pytest.raises(ValueError):
        CLAM_SB(in_shape=(10,), inst_loss_name="InvalidLoss")


def test_clam_sb_rejects_multiclass(features, mask):
    model_binary = CLAM_SB(in_shape=(10,), n_outputs=1)
    assert model_binary.num_outputs == 1
    Y_pred = model_binary(features, mask)
    assert Y_pred.shape == (2,)

    with pytest.raises(ValueError, match="only supports binary classification"):
        CLAM_SB(in_shape=(10,), n_outputs=3)
