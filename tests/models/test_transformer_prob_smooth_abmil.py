import torch

from torchmil.models.transformer_prob_smooth_abmil import (
    TransformerProbSmoothABMIL,
)  # Assuming your script is saved as 'your_module.py'


# Helper function to create dummy data
def create_dummy_data(batch_size=2, bag_size=3, feat_dim=5):
    X = torch.randn(batch_size, bag_size, feat_dim)
    Y = torch.randint(0, 2, (batch_size,))
    adj = torch.eye(bag_size).unsqueeze(0).repeat(batch_size, 1, 1)
    mask = torch.ones(batch_size, bag_size, dtype=torch.bool)
    return X, Y, adj, mask


class TestTransformerProbSmoothABMIL:
    def test_forward(self):
        model = TransformerProbSmoothABMIL(in_shape=(5,))
        model.eval()
        X, _, adj, mask = create_dummy_data()
        output = model(X, adj)
        assert output.shape == (X.shape[0],)

        output_with_att = model(X, adj, return_att=True)
        assert isinstance(output_with_att, tuple)
        assert len(output_with_att) == 2
        assert output_with_att[0].shape == (X.shape[0],)
        assert output_with_att[1].shape == (X.shape[0], X.shape[1])

        output_with_samples_att = model(X, adj, return_att=True, return_samples=True)
        assert isinstance(output_with_samples_att, tuple)
        assert len(output_with_samples_att) == 2
        assert output_with_samples_att[0].shape == (
            X.shape[0],
            model.pool.n_samples_test,
        )
        assert output_with_samples_att[1].shape == (
            X.shape[0],
            X.shape[1],
            model.pool.n_samples_test,
        )

        output_with_kl = model(X, adj, return_kl_div=True)
        assert isinstance(output_with_kl, tuple)
        assert len(output_with_kl) == 2
        assert output_with_kl[0].shape == (X.shape[0],)
        assert isinstance(output_with_kl[1], torch.Tensor)

        output_with_all = model(
            X, adj, mask, return_att=True, return_samples=True, return_kl_div=True
        )
        assert isinstance(output_with_all, tuple)
        assert len(output_with_all) == 3
        assert output_with_all[0].shape == (X.shape[0], model.pool.n_samples_test)
        assert output_with_all[1].shape == (
            X.shape[0],
            X.shape[1],
            model.pool.n_samples_test,
        )
        assert isinstance(output_with_all[2], torch.Tensor)

    def test_compute_loss(self):
        model = TransformerProbSmoothABMIL(in_shape=(5,))
        X, Y, adj, mask = create_dummy_data()
        Y_pred, loss_dict = model.compute_loss(Y, X, adj, mask)
        assert Y_pred.shape == (X.shape[0],)
        assert isinstance(loss_dict, dict)
        assert "BCEWithLogitsLoss" in loss_dict
        assert "KLDiv" in loss_dict
        assert isinstance(loss_dict["BCEWithLogitsLoss"], torch.Tensor)
        assert isinstance(loss_dict["KLDiv"], torch.Tensor)

    def test_predict(self):
        model = TransformerProbSmoothABMIL(in_shape=(5,))
        model.eval()
        X, _, adj, mask = create_dummy_data()
        Y_pred, att_val = model.predict(X, adj, mask)
        assert Y_pred.shape == (X.shape[0],)
        assert att_val.shape == (X.shape[0], X.shape[1])

        Y_pred_no_inst = model.predict(X, adj, mask, return_inst_pred=False)
        assert Y_pred_no_inst.shape == (X.shape[0],)

        Y_pred_samples, att_val_samples = model.predict(
            X, adj, mask, return_inst_pred=True, return_samples=True
        )
        assert Y_pred_samples.shape == (X.shape[0], model.pool.n_samples_test)
        assert att_val_samples.shape == (
            X.shape[0],
            X.shape[1],
            model.pool.n_samples_test,
        )

    def test_multiclass_support(self):
        """Test multiclass support for TransformerProbSmoothABMIL model."""
        X, _, adj, mask = create_dummy_data()

        # Binary classification (n_outputs=1)
        model_binary = TransformerProbSmoothABMIL(in_shape=(5,), n_outputs=1)
        assert model_binary.num_outputs == 1
        Y_pred = model_binary(X, adj)
        assert Y_pred.shape == (X.shape[0],)

        # 3-class classification (n_outputs=3)
        model_3class = TransformerProbSmoothABMIL(in_shape=(5,), n_outputs=3)
        assert model_3class.num_outputs == 3
        Y_pred = model_3class(X, adj)
        assert Y_pred.shape == (X.shape[0], 3)

        # 5-class classification (n_outputs=5)
        model_5class = TransformerProbSmoothABMIL(in_shape=(5,), n_outputs=5)
        assert model_5class.num_outputs == 5
        Y_pred = model_5class(X, adj)
        assert Y_pred.shape == (X.shape[0], 5)
