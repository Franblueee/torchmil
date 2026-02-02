import torch

from .mil_model import MILModel
from torchmil.nn import AttentionPool
from torchmil.nn.utils import get_feat_dim, LazyLinear, SmoothTop1SVM


class CLAM_SB(MILModel):
    r"""

    Clustering-constrained Attention Multiple Instance Learning (CLAM), proposed in the paper [Data Efficient and Weakly Supervised Computational Pathology on Whole Slide Images](https://arxiv.org/abs/2004.09666).

    **Overview.**
    The forward pass of CLAM is identical to the forward pass of [ABMIL](./abmil.md). The difference lies in the instance-level regularization, which we describe below.

    **Instance-level regularization.**
    CLAM uses a binary clustering objective during training.
    For this, in the binary MIL setting, two clustering classifiers are considered: $c_0 \colon \mathbb{R}^D \to \mathbb{R}$ and $c_1 \colon \mathbb{R}^D \to \mathbb{R}$.
    To supervise this objective, the attention values computes by the attention pooling are used to generate *pseudo labels*.

    Given an input bag $\mathbf{X} = \left[ \mathbf{x}_1, \ldots, \mathbf{x}_N \right]^\top \in \mathbb{R}^{N \times P}$ with label $Y$ and attention values $\mathbf{f} = \left[ f_1, \ldots, f_N \right]^\top \in \mathbb{R}^{N}$,
    the instance-level regularization is performed as follows:

    1. The $k$ instances with the highest attention values are selected as in-the-class instances. The $k$ instances with the lowest attention values are selected as out-of-the-class instances,

        \begin{gather}
            D_{\text{in}} = \left\{ \mathbf{x}_i \mid f_i \in \text{TopK}(\mathbf{f}, k) \right\}, \\
            D_{\text{out}} = \left\{ \mathbf{x}_i \mid f_i \in \text{BottomK}(\mathbf{f}, k) \right\}.
        \end{gather}

    2. The instances in $D_{\text{in}}$ are assigned a pseudo label of 1 for $c_Y$, and a pseudo label of 0 for $c_{1-Y}$.
    The instances in $D_{\text{out}}$ are assigned a pseudo label of 0 for $c_Y$. The pseudo labels are used to train the clustering classifiers,

        \begin{gather}
            \ell_{\text{in}} = \frac{1}{2K} \left( \sum_{\mathbf{x} \in D_{\text{in}}}\ell_{\text{inst}}(c_Y(\mathbf{x}), 1) + \sum_{\mathbf{x} \in D_{\text{out}}}\ell_{\text{inst}}(c_{Y}(\mathbf{x}), 0) \right), \\
            \ell_{\text{out}} = \frac{1}{K} \sum_{\mathbf{x} \in D_{\text{in}}}\ell_{\text{inst}}(c_{1-Y}(\mathbf{x}), 0),
        \end{gather}

    where $\ell_{\text{inst}}$ is the instance-level loss function (the default is [SmoothTop1SVM](https://jmlr.csail.mit.edu/papers/volume2/crammer01a/crammer01a.pdf)) and $Y$ is the true bag label.
    The total instance-level loss is $\ell_{\text{in}} + \ell_{\text{out}}$, which is added to the bag-level loss to train the model.

    """

    def __init__(
        self,
        in_shape: tuple = None,
        att_dim: int = 128,
        att_act: str = "tanh",
        k_sample: int = 10,
        gated: bool = False,
        inst_loss_name: str = "SmoothTop1SVM",
        feat_ext: torch.nn.Module = torch.nn.Identity(),
        criterion: torch.nn.Module = torch.nn.BCEWithLogitsLoss(),
    ) -> None:
        """
        Arguments:
            in_shape: Shape of input data expected by the feature extractor (excluding batch dimension).
            att_dim: Attention dimension.
            att_act: Activation function for attention. Possible values: 'tanh', 'relu', 'gelu'.
            k_sample: Number of instances to sample.
            gated: If True, use gated attention in the attention pooling.
            feat_ext: Feature extractor.
            criterion: Loss function. By default, Binary Cross-Entropy loss from logits.
        """
        super().__init__()
        self.criterion = criterion
        self.feat_ext = feat_ext
        self.k_sample = k_sample

        if in_shape is not None:
            feat_dim = get_feat_dim(feat_ext, in_shape)
        else:
            feat_dim = None

        self.pool = AttentionPool(
            in_dim=feat_dim, att_dim=att_dim, act=att_act, gated=gated
        )
        self.classifier = LazyLinear(feat_dim, 1)
        self.inst_classifiers = torch.nn.ModuleList(
            [LazyLinear(feat_dim, 2) for i in range(2)]
        )
        if inst_loss_name == "SmoothTop1SVM":
            self.inst_loss_fn = SmoothTop1SVM(n_classes=2)
        elif inst_loss_name == "BCEWithLogitsLoss":
            self.inst_loss_fn = torch.nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Invalid instance loss name: {inst_loss_name}")

    @staticmethod
    def create_positive_targets(length: int, device: torch.device) -> torch.Tensor:
        """
        Create positive targets.

        Arguments:
            length: Length of the target tensor.
            device: Device to create the tensor.

        Returns:
            pos_targets: Tensor of shape `(length,)` with all elements set to 1.
        """
        return torch.full((length,), 1, device=device).long()

    @staticmethod
    def create_negative_targets(length: int, device: torch.device) -> torch.Tensor:
        """
        Create negative targets.

        Arguments:
            length: Length of the target tensor.
            device: Device to create the tensor.

        Returns:
            neg_targets: Tensor of shape `(length,)` with all elements set to 0.
        """
        return torch.full((length,), 0, device=device).long()

    def inst_eval(
        self,
        att: torch.Tensor,
        emb: torch.Tensor,
        classifier: torch.nn.Module,
        mask=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate instance-level loss for in-the-class attention branch.

        Arguments:
            att: Attention values of shape `(bag_size,)`.
            emb: Embeddings of shape `(bag_size, feat_dim)`.
            classifier: Instance classifier.
            mask: Mask of shape `(bag_size,)`. If provided, the attention values are masked before selection.

        Returns:
            instance_loss: Instance loss.
            all_preds: Predicted instance labels of shape `(2 * k_sample,)`.
            all_targets: Generated instance labels of shape `(2 * k_sample,)`.
        """
        device = att.device
        bag_size = emb.shape[0]

        if mask is not None:
            bag_size = mask.sum().item()
            mask = mask.bool()

        k_sample = min(self.k_sample, bag_size)

        if mask is not None:
            att = att.masked_fill(~mask, -1e9)  # (bag_size,)
        top_p_ids = torch.topk(att, k_sample)[1]  # (k_sample,)
        top_p = torch.index_select(emb, dim=0, index=top_p_ids)  # (k_sample, feat_dim)

        if mask is not None:
            att = att.masked_fill(~mask, 1e9)  # (bag_size,)

        top_n_ids = torch.topk(-att, k_sample)[1]  # (k_sample,)
        top_n = torch.index_select(emb, dim=0, index=top_n_ids)  # (k_sample, feat_dim)

        p_targets = self.create_positive_targets(k_sample, device)  # (k_sample,)
        n_targets = self.create_negative_targets(k_sample, device)  # (k_sample,)

        all_targets = torch.cat([p_targets, n_targets], dim=0)  # (2 * k_sample,)
        all_instances = torch.cat([top_p, top_n], dim=0)  # (2 * k_sample, feat_dim)
        logits = classifier(all_instances)  # (2 * k_sample, 2)
        all_preds = torch.topk(logits, 1, dim=1)[1]  # (2 * k_sample,)
        instance_loss = self.inst_loss_fn(
            logits.float(), all_targets.unsqueeze(-1).float()
        )
        return instance_loss, all_preds, all_targets

    def inst_eval_out(
        self,
        att: torch.Tensor,
        emb: torch.Tensor,
        classifier: torch.nn.Module,
        mask=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate instance-level loss for out-of-the-class attention branch.

        Arguments:
            att: Attention values of shape `(bag_size,)`.
            emb: Embeddings of shape `(bag_size, feat_dim)`.
            classifier: Instance classifier.
            mask: Mask of shape `(bag_size,)`. If provided, the attention values are masked before selection.

        Returns:
            instance_loss: Instance loss.
            p_preds: Predicted instance labels of shape `(k_sample,)`.
            p_targets: Generated instance labels of shape `(k_sample,)`.
        """
        device = att.device
        bag_size = emb.shape[0]

        if mask is not None:
            bag_size = mask.sum().item()
            mask = mask.bool()

        k_sample = min(self.k_sample, bag_size)

        if mask is not None:
            att = att.masked_fill(~mask, -1e9)

        top_p_ids = torch.topk(att, k_sample)[1]  # (k_sample,)
        top_p = torch.index_select(emb, dim=0, index=top_p_ids)  # (k_sample, feat_dim)
        p_targets = self.create_negative_targets(k_sample, device)  # (k_sample,)
        logits = classifier(top_p)  # (k_sample, 2)
        p_preds = torch.topk(logits, 1, dim=1)[1]  # (k_sample,)
        instance_loss = self.inst_loss_fn(
            logits.float(), p_targets.unsqueeze(-1).float()
        )  # (k_sample,)
        return instance_loss, p_preds, p_targets

    def compute_inst_loss(
        self,
        att: torch.Tensor,
        emb: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Computes instance loss.

        Arguments:
            att: Attention values of shape `(batch_size, bag_size)`.
            emb: Embeddings of shape `(batch_size, bag_size, feat_dim)`.
            labels: Bag labels of shape `(batch_size,)`.
            mask: Mask of shape `(batch_size, bag_size)`. If provided, the attention values are masked before selection.

        Returns:
            inst_loss: Instance loss.
        """

        sum_inst_loss = 0.0
        batch_size = att.shape[0]

        for i in range(batch_size):
            label = int(labels[i].item())
            if label == 0:
                in_idx = 0
                out_idx = 1
            else:
                in_idx = 1
                out_idx = 0
            inst_loss_in, _, _ = self.inst_eval(
                att[i],
                emb[i],
                self.inst_classifiers[in_idx],
                mask[i] if mask is not None else None,
            )
            inst_loss_out, _, _ = self.inst_eval_out(
                att[i],
                emb[i],
                self.inst_classifiers[out_idx],
                mask[i] if mask is not None else None,
            )

            sum_inst_loss += inst_loss_in + inst_loss_out
        return sum_inst_loss

    def forward(
        self,
        X: torch.Tensor,
        mask: torch.Tensor = None,
        return_att: bool = False,
        return_emb: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.

        Arguments:
            X: Bag features of shape `(batch_size, bag_size, ...)`.
            mask: Mask of shape `(batch_size, bag_size)`.
            return_att: If True, returns attention values (before normalization) in addition to `Y_pred`.
            return_emb: If True, returns embeddings in addition to `Y_pred`.

        Returns:
            Y_pred: Bag label logits of shape `(batch_size,)`.
            att: Only returned when `return_att=True`. Attention values (before normalization) of shape (batch_size, bag_size).
            emb: Only returned when `return_emb=True`. Embeddings of shape (batch_size, bag_size, feat_dim).
        """

        X = self.feat_ext(X)  # (batch_size, bag_size, D)

        z, f = self.pool(
            X, mask, return_att=True
        )  # z: (batch_size, D), f: (batch_size, bag_size)

        Y_pred = self.classifier(z).squeeze(1)  # (batch_size,)

        if return_emb:
            if return_att:
                return Y_pred, f, X
            else:
                return Y_pred, X
        elif return_att:
            return Y_pred, f
        else:
            return Y_pred

    def compute_loss(
        self, Y: torch.Tensor, X: torch.Tensor, mask: torch.Tensor = None
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute loss given true bag labels.

        Arguments:
            Y: Bag labels of shape `(batch_size,)`.
            X: Bag features of shape `(batch_size, bag_size, ...)`.
            mask: Mask of shape `(batch_size, bag_size)`.

        Returns:
            Y_pred: Bag label logits of shape `(batch_size,)`.
            loss_dict: Dictionary containing the loss value.
        """
        Y_pred, att, emb = self.forward(X, mask, return_att=True, return_emb=True)
        crit_loss = self.criterion(Y_pred.float(), Y.float())
        crit_name = self.criterion.__class__.__name__
        inst_loss = self.compute_inst_loss(att, emb, Y)

        return Y_pred, {crit_name: crit_loss, "InstLoss": inst_loss}

    def predict(
        self, X: torch.Tensor, mask: torch.Tensor = None, return_inst_pred: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict bag labels.

        Arguments:
            X: Bag features of shape `(batch_size, bag_size, ...)`.
            mask: Mask of shape `(batch_size, bag_size)`.

        Returns:
            Y_pred: Predicted bag labels of shape `(batch_size,)`.
            y_inst_pred: Predicted instance labels of shape `(batch_size, bag_size)`. Only returned when `return_inst_pred=True`.
        """
        return self.forward(X, mask, return_att=return_inst_pred)
