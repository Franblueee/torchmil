import numpy as np
import torch
import warnings

from torchmil.datasets import ProcessedMILDataset


class BinaryClassificationDataset(ProcessedMILDataset):
    r"""
    Dataset for binary classification MIL problems. See [`torchmil.datasets.ProcessedMILDataset`](./processed_mil_dataset.md) for more information.

    For a given bag with bag label $Y$ and instance labels $\left\{ y_1, \ldots, y_N \right \}$, this dataset assumes that

    \begin{gather}
        Y \in \left\{ 0, 1 \right\}, \quad y_n \in \left\{ 0, 1 \right\}, \quad \forall n \in \left\{ 1, \ldots, N \right\},\\
        Y = \max \left\{ y_1, \ldots, y_N \right\}.
    \end{gather}

    When the instance labels are not provided, they are set to 0 if the bag label is 0, and to -1 if the bag label is 1.
    If the instance labels are provided, but they are not consistent with the bag label, a warning is issued and the instance labels are all set to -1.
    """

    def __init__(
        self,
        features_path: str,
        labels_path: str,
        inst_labels_path: str = None,
        coords_path: str = None,
        bag_names: list = None,
        bag_keys: list = ["X", "Y", "y_inst", "adj", "coords"],
        dist_thr: float = 1.5,
        adj_with_dist: bool = False,
        norm_adj: bool = True,
        load_at_init: bool = True,
        verbose: bool = True,
    ) -> None:
        super().__init__(
            features_path=features_path,
            labels_path=labels_path,
            inst_labels_path=inst_labels_path,
            coords_path=coords_path,
            bag_names=bag_names,
            bag_keys=bag_keys,
            dist_thr=dist_thr,
            adj_with_dist=adj_with_dist,
            norm_adj=norm_adj,
            load_at_init=load_at_init,
            verbose=verbose,
        )

    def _fix_inst_labels(self, inst_labels):
        """
        Make sure that instance labels have shape (bag_size,).
        """
        if inst_labels is not None:
            while inst_labels.ndim > 1:
                inst_labels = np.squeeze(inst_labels, axis=-1)
        return inst_labels

    def _fix_labels(self, labels):
        """
        Make sure that labels have shape ().
        """
        labels = np.squeeze(labels)
        return labels

    def _load_inst_labels(self, name):
        inst_labels = super()._load_inst_labels(name)
        inst_labels = self._fix_inst_labels(inst_labels)
        return inst_labels

    def _load_labels(self, name):
        labels = super()._load_labels(name)
        labels = self._fix_labels(labels)
        return labels

    def _consistency_check(self, bag_dict, name):
        """
        Check if the instance labels are consistent with the bag label.
        """
        if "Y" in bag_dict:
            if "y_inst" in bag_dict:
                if bag_dict["Y"] != (bag_dict["y_inst"]).max():
                    if self.verbose:
                        msg = f"Instance labels (max(y_inst)={(bag_dict['y_inst']).max()}) are not consistent with bag label (Y={bag_dict['Y']}) for bag {name}. Setting all instance labels to -1 (unknown)."
                        warnings.warn(msg)
                    bag_dict["y_inst"] = np.full((bag_dict["X"].shape[0],), -1)
            else:
                if bag_dict["Y"] == 0:
                    bag_dict["y_inst"] = np.zeros(bag_dict["X"].shape[0])
                else:
                    if self.verbose:
                        msg = f"Instance labels not found for bag {name}. Setting all to -1."
                        warnings.warn(msg)
                    bag_dict["y_inst"] = np.full((bag_dict["X"].shape[0],), -1)
        return bag_dict

    def _load_bag(self, name: str) -> dict[str, torch.Tensor]:
        bag_dict = super()._load_bag(name)
        bag_dict = self._consistency_check(bag_dict, name)
        return bag_dict
