import torch
import numpy as np

from torchmil.models import MILModel
from torchmil.nn import AttentionPool, LazyLinear
from torchmil.nn.utils import get_feat_dim


class DTFDMIL(MILModel):
    r"""
    Double-Tier Feature Distillation Multiple Instance Learning (DFTD-MIL) model, proposed in the paper [DTFD-MIL: Double-Tier Feature Distillation Multiple Instance Learning for Histopathology Whole Slide Image Classification](https://arxiv.org/abs/2203.12081).

    **Overview.**
    Given an input bag $\mathbf{X} = \left[ \mathbf{x}_1, \ldots, \mathbf{x}_N \right]^\top \in \mathbb{R}^{N \times P}$, the model optionally applies a feature extractor, $\text{FeatExt}(\cdot)$, to transform the instance features: $\mathbf{X} = \text{FeatExt}(\mathbf{X}) \in \mathbb{R}^{N \times D}$.

    Then, the instances in a bag are randomly grouped in $M$ pseudo-bags $\{\mathbf{X}_1, \cdots, \mathbf{X}_M\}$ with approximately the same number of instances. Each pseudo-bag is assigned its parent's bag label $Y_m = Y$. Then, the model has two prediction tiers:

    In **Tier 1**, the model uses the attention pool (see [AttentionPool](../nn/attention/attention_pool.md) for details) and a classifier, jointly noted as $T_1$ to predict the label of each pseudo-bag,

    $$ \widehat{Y}_m = T_1(\mathbf{X}_m).$$

    The loss associated to this tier is the binary cross entropy computed using the pseudo-bag labels $Y_m$ and the predicted label $\widehat{Y}_m$.

    In **Tier 2**, Grad-CAM (see [Grad-CAM](https://arxiv.org/abs/1610.02391) for details) is used to compute the probability of each instance. Based on that probability, a feature vector $\mathbf{z}^m$ is distilled for the $m$-th pseudo-bag. Then, the model uses another attention pool and a classifier, jointly noted as $T_2$ to predict the final label of the bag,

    $$ \widehat{Y} = T_2\left( \left[ \mathbf{z}_1, \ldots, \mathbf{z}_M \right]^\top  \right).$$

    The loss associated to this tier is the binary cross entropy computed using the bag labels $Y$ and the predicted label $\widehat{Y}$.

    **Loss function.**
    By default, the model is trained end-to-end using the followind per-bag loss:

    $$ \ell = \ell_{\text{BCE}}(Y, \widehat{Y}) + \frac{1}{M} \sum_{m=1}^{M} \ell_{\text{BCE}}(Y_m, \widehat{Y}_m),$$

    where $\ell_{\text{BCE}}$ is the binary cross entropy loss.

    """

    def __init__(
        self,
        in_shape: tuple = None,
        att_dim: int = 128,
        n_groups: int = 8,
        distill_mode: str = "maxmin",
        feat_ext: torch.nn.Module = torch.nn.Identity(),
        n_outputs: int = 1,
        criterion: torch.nn.Module = torch.nn.BCEWithLogitsLoss(),
    ) -> None:
        """
        Arguments:
            in_shape: Shape of input data expected by the feature extractor (excluding batch dimension). If not provided, it will be lazily initialized.
            att_dim: Attention dimension.
            n_groups: Number of groups to split the bag instances.
            distill_mode: Distillation mode. Possible values: 'maxmin', 'max', 'afs'.
            feat_ext: Feature extractor.
            n_outputs: Number of outputs. DTFDMIL only supports 1 output for binary classification.
            criterion: Loss function. By default, Binary Cross-Entropy loss from logits.
        """

        super(DTFDMIL, self).__init__()
        if n_outputs > 1:
            raise ValueError("DTFDMIL only supports binary classification (n_outputs=1).")
        self.feat_ext = feat_ext
        self.criterion = criterion
        self.n_groups = n_groups
        self.distill_mode = distill_mode

        if distill_mode not in ["maxmin", "max", "afs"]:
            raise ValueError(
                f"Invalid distill_mode: {distill_mode}. Choose from ['maxmin', 'max', 'afs']"
            )

        if in_shape is not None:
            feat_dim = get_feat_dim(feat_ext, in_shape)
        else:
            feat_dim = None

        self.attention_pool = AttentionPool(in_dim=feat_dim, att_dim=att_dim)
        self.classifier = LazyLinear(feat_dim, 1)

        self.u_attention_pool = AttentionPool(in_dim=feat_dim, att_dim=att_dim)
        self.u_classifier = LazyLinear(feat_dim, 1)

    def _cam_1d(self, classifier, features):
        tweight = list(classifier.parameters())[-2]
        cam_maps = torch.einsum("bgf,cf->bcg", [features, tweight])
        return cam_maps

    def forward(
        self,
        X: torch.Tensor,
        mask: torch.Tensor = None,
        return_pseudo_pred: bool = False,
        return_inst_cam: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Arguments:
            X: Bag features of shape `(batch_size, bag_size, ...)`.
            mask: Mask of shape `(batch_size, bag_size)`.
            return_pseudo_pred: If True, returns pseudo label logits in addition to `Y_pred`.
            return_inst_cam: If True, returns instance-level CAM values in addition to `Y_pred`.

        Returns:
            Y_pred: Bag label logits of shape `(batch_size,)`.
            inst_cam: Only returned when `return_inst_cam=True`. Instance-level CAM values of shape (batch_size, bag_size).
        """

        batch_size = X.size(0)
        bag_size = X.size(1)

        if bag_size < self.n_groups:
            n_groups = bag_size
        else:
            n_groups = self.n_groups

        bag_index = np.arange(0, bag_size)
        np.random.shuffle(bag_index)
        bag_chunks = np.array_split(bag_index, n_groups)

        X = self.feat_ext(X)  # (batch_size, bag_size, feat_dim)
        feat_dim = X.size(-1)

        pseudo_pred_list = []
        pseudo_feat_list = []
        inst_cam_list = []
        for bag_chunk in bag_chunks:
            X_chunk = X[:, bag_chunk, :]
            mask_chunk = mask[:, bag_chunk].bool() if mask is not None else None
            chunk_size = X_chunk.size(1)

            z = self.attention_pool(
                X_chunk, mask_chunk
            )  # (batch_size, feat_dim), [batch_size, chunk_size)

            pseudo_pred = self.classifier(z)  # (batch_size, n_outputs)
            pseudo_pred_list.append(pseudo_pred)

            inst_cam = self._cam_1d(
                self.classifier, X_chunk
            )  # (batch_size, 1, chunk_size)
            inst_cam = inst_cam.squeeze(1)  # (batch_size, chunk_size)
            inst_cam_list.append(inst_cam)

            if self.distill_mode == "afs":
                pseudo_feat = z.unsqueeze(1)  # (batch_size, 1, feat_dim)
            else:
                inst_cam_max = inst_cam.masked_fill(
                    ~mask_chunk, -1e9
                )  # (batch_size, chunk_size)
                inst_cam_min = inst_cam.masked_fill(
                    ~mask_chunk, 1e9
                )  # (batch_size, chunk_size)

                sort_idx_max = torch.sort(inst_cam_max, 1, descending=True)[
                    1
                ]  # (batch_size, chunk_size)
                topk_idx_max = sort_idx_max[
                    :, :chunk_size
                ].long()  # (batch_size, chunk_size)

                sort_idx_min = torch.sort(inst_cam_min, 1, descending=False)[
                    1
                ]  # (batch_size, chunk_size)
                topk_idx_min = sort_idx_min[
                    :, :chunk_size
                ].long()  # (batch_size, chunk_size)

                topk_idx = torch.cat(
                    [topk_idx_max, topk_idx_min], dim=1
                )  # (batch_size, 2*chunk_size)

                if self.distill_mode == "maxmin":
                    index = topk_idx.unsqueeze(-1).expand(
                        -1, -1, feat_dim
                    )  # (batch_size, 2*chunk_size, feat_dim)
                    pseudo_feat = torch.gather(
                        X_chunk, 1, index
                    )  # (batch_size, 2*chunk_size, feat_dim)
                elif self.distill_mode == "max":
                    index = topk_idx_max.unsqueeze(-1).expand(
                        -1, -1, feat_dim
                    )  # (batch_size, chunk_size, feat_dim)
                    pseudo_feat = torch.gather(
                        X_chunk, 1, index
                    )  # (batch_size, chunk_size, feat_dim)

            pseudo_feat_list.append(pseudo_feat)

        # Concatenate pseudo predictions from all groups
        pseudo_pred = torch.cat(pseudo_pred_list, dim=1)  # (batch_size, n_groups)
        pseudo_feat = torch.cat(
            pseudo_feat_list, dim=1
        )  # (batch_size, n_groups, k, feat_dim); k = 2*chunk_size or chunk_size or 1
        pseudo_feat = pseudo_feat.view(
            batch_size, -1, feat_dim
        )  # (batch_size, n_groups*k, feat_dim)

        pseudo_z = self.u_attention_pool(pseudo_feat)  # (batch_size, feat_dim)
        Y_pred = self.u_classifier(pseudo_z)  # (batch_size, 1)
        Y_pred = Y_pred.squeeze(-1)  # (batch_size,)

        if return_inst_cam:
            inst_cam = torch.cat(inst_cam_list, dim=1)  # (batch_size, bag_size)

            inst_cam_reorder = torch.zeros_like(inst_cam)
            bag_chunks_idx = np.concatenate(bag_chunks)
            inst_cam_reorder[:, bag_chunks_idx] = inst_cam
            inst_cam = inst_cam_reorder

            if return_pseudo_pred:
                return Y_pred, pseudo_pred, inst_cam
            else:
                return Y_pred, inst_cam
        else:
            if return_pseudo_pred:
                return Y_pred, pseudo_pred
            else:
                return Y_pred

    def compute_loss(
        self,
        Y: torch.Tensor,
        X: torch.Tensor,
        mask: torch.Tensor = None,
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
        Y_pred, pseudo_pred = self.forward(
            X, mask, return_pseudo_pred=True
        )  # (batch_size,], [batch_size, n_groups]
        n_groups = pseudo_pred.size(1)
        crit_name = self.criterion.__class__.__name__
        crit_loss_t1 = self.criterion(Y_pred, Y.float())
        Y_repeat = Y.unsqueeze(1).repeat(1, n_groups)  # (batch_size, num_groups]
        crit_loss_t2 = self.criterion(pseudo_pred, Y_repeat.float()).mean()
        return Y_pred, {
            f"{crit_name}_t1": crit_loss_t1,
            f"{crit_name}_t2": crit_loss_t2,
        }

    def predict(
        self,
        X: torch.Tensor,
        mask: torch.Tensor = None,
        return_inst_pred: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict bag and (optionally) instance labels.

        Arguments:
            X: Bag features of shape `(batch_size, bag_size, ...)`.
            mask: Mask of shape `(batch_size, bag_size)`.
            return_inst_pred: If `True`, returns instance labels predictions, in addition to bag label predictions.

        Returns:
            Y_pred: Bag label logits of shape `(batch_size,)`.
            y_inst_pred: If `return_inst_pred=True`, returns instance labels predictions of shape `(batch_size, bag_size)`.
        """
        return self.forward(X, mask, return_inst_cam=return_inst_pred)
