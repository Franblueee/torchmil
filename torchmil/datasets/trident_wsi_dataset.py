import numpy as np
import os

from .processed_mil_dataset import ProcessedMILDataset
import pandas as pd
import h5py


class TridentWSIDataset(ProcessedMILDataset):
    r"""
    This class represents a dataset of Whole Slide Images (WSI) for Multiple Instance Learning (MIL) that was processed using the [TRIDENT](https://github.com/mahmoodlab/TRIDENT) repository.

    **Directory structure.**
    For more information on the processing of the bags, refer to the [`ProcessedMILDataset` class](processed_mil_dataset.md).
    This dataset expects the directory structure provided by the TRIDENT repository. The `base_path` argument should point to the base directory of the TRIDENT output, of the form `{mag}x_{ps}px_{opx}px_overlap/`. In this folder, the following folders are expected:
    ```
    base_path
    ├──features_{feature_extractor}
    |   ├── wsi1.h5
    |   ├── wsi2.h5
    |   └── ...
    ├──patches
    |   ├── wsi1_patches.h5
    |   ├── wsi2_patches.h5
    |   └── ...
    └──patch_labels (optional)
        ├── wsi1.h5
        ├── wsi2.h5
        └── ...
    ```

    **Adjacency matrix.**
    If the coordinates of the patches are available, an adjacency matrix representing the spatial relationships between the patches is built. Please refer to the [`ProcessedMILDataset` class](processed_mil_dataset.md) for more information on how the adjacency matrix is built.

    **WSI-level labels.**
    The labels of the WSIs can be provided in two ways:
    1. As a directory containing one file per WSI, following the same structure as the features and patches folders.
    2. As a CSV file containing the WSI names and their corresponding labels. In this case, the user must provide the column names for the WSI names and labels using the `wsi_name_col` and `wsi_label_col` keyword arguments, respectively.

    **Patch-level labels.**
    The labels of the patches can be provided through the `patch_labels_path` argument. This should be a directory containing one '.h5' file per WSI. This file should have "patch_labels" as a key, which should contain an array with the labels of the patches. The order of the patch labels should be the same as the order of the features and coordinates of the patches.
    """

    def __init__(
        self,
        base_path: str,
        labels_path: str,
        feature_extractor: str,
        patch_labels_path: str = None,
        wsi_names: list = None,
        bag_keys: list = ["X", "Y", "y_inst", "adj", "coords"],
        patch_size: int = 512,
        dist_thr: float = None,
        adj_with_dist: bool = False,
        norm_adj: bool = True,
        load_at_init: bool = True,
        wsi_name_col: str = None,
        wsi_label_col: str = None,
    ) -> None:
        """
        Class constructor.

        Arguments:
            base_path: Path to the base directory containing the TRIDENT folders.
            labels_path: Path to the directory or CSV file containing the labels of the WSIs.
            feature_extractor: Feature extractor used to extract the features. This will determine the features folder name.
            patch_labels_path: Path to the directory containing the labels of the patches.
            wsi_names: List of the names of the WSIs to load. If None, all the WSIs in the `features_path` directory are loaded.
            bag_keys: List of keys to use for the bags. Must be in ['X', 'Y', 'y_inst', 'coords'].
            patch_size: Size of the patches.
            dist_thr: Distance threshold for building the adjacency matrix. If None, it is set to `sqrt(2) * patch_size`.
            adj_with_dist: If True, the adjacency matrix is built using the Euclidean distance between the patches features. If False, the adjacency matrix is binary.
            norm_adj: If True, normalize the adjacency matrix.
            load_at_init: If True, load the bags at initialization. If False, load the bags on demand.
            wsi_name_col: Name of the column containing the WSI names in the CSV file provided in `labels_path`. Only used if `labels_path` is a CSV file.
            wsi_label_col: Name of the column containing the WSI labels in the CSV file provided in `labels_path`. Only used if `labels_path` is a CSV file.
        """
        if dist_thr is None:
            # dist_thr = np.sqrt(2.0) * patch_size
            dist_thr = np.sqrt(2.0)

        self.base_path = base_path
        self.feature_extractor = feature_extractor
        features_path = self.base_path + f"features_{feature_extractor}/"
        coords_path = self.base_path + "patches/"

        self.patch_size = patch_size
        self.wsi_name_col = wsi_name_col
        self.wsi_label_col = wsi_label_col

        super().__init__(
            features_path=features_path,
            labels_path=labels_path,
            inst_labels_path=patch_labels_path,
            coords_path=coords_path,
            bag_names=wsi_names,
            bag_keys=bag_keys,
            file_ext=".h5",
            dist_thr=dist_thr,
            adj_with_dist=adj_with_dist,
            norm_adj=norm_adj,
            load_at_init=load_at_init,
        )

    def _load_coords(self, name: str) -> np.ndarray:
        """
        Load the coordinates of a bag from disk.

        Arguments:
            name: Name of the bag to load.

        Returns:
            coords: Coordinates of the bag.
        """
        coords_file = os.path.join(self.coords_path, name + "_patches" + self.file_ext)
        coords = h5py.File(coords_file, "r")["coords"][:]
        if coords is not None:
            coords = coords / self.patch_size
            coords = coords.astype(int)
        return coords

    def _load_inst_labels(self, name: str) -> np.ndarray:
        """
        Load the instance labels of a bag from disk.

        Arguments:
            name: Name of the bag to load.

        Returns:
            inst_labels: Instance labels of the bag.
        """
        inst_labels_file = os.path.join(self.inst_labels_path, name + self.file_ext)
        inst_labels = self._read_file(inst_labels_file, "patch_labels")
        return inst_labels

    def _load_labels(self, name: str) -> np.ndarray:
        """
        Load the labels of a bag from disk.
        Allows to load the labels from a CSV file, where the WSI names and labels are provided in two columns.
        The column names for the WSI names and labels should be provided through the `wsi_name_col` and `wsi_label_col` keyword arguments, respectively.

        Arguments:
            name: Name of the bag to load.

        Returns:
            labels: Labels of the bag.
        """

        if os.path.isdir(self.labels_path):
            return super()._load_labels(name)
        else:
            if not hasattr(self, "labels_csv"):
                self.labels_csv = pd.read_csv(os.path.join(self.labels_path))
            if self.wsi_name_col is not None and self.wsi_label_col is not None:
                try:
                    self.labels_csv[self.wsi_name_col] = self.labels_csv[
                        self.wsi_name_col
                    ].apply(lambda x: os.path.splitext(x)[0])
                    label = self.labels_csv.loc[
                        self.labels_csv[self.wsi_name_col] == name, self.wsi_label_col
                    ].values
                except ValueError:
                    raise ValueError(
                        f"Could not read the label of the file {name} from the CSV file {self.labels_path}. Please check that the column names provided in 'wsi_name_col' and 'wsi_label_col' are correct."
                    )
            else:
                raise ValueError(
                    "When providing a CSV file for labels_path, you must provide the column names for the WSI names and labels using the 'wsi_name_col' and 'wsi_label_col' arguments, respectively."
                )
            label = np.array(label)
            return label
