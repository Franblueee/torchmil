import numpy as np
import os

from .wsi_dataset import WSIDataset
import pandas as pd
import h5py


class TridentWSIDataset(WSIDataset):
    r"""
    This class represents a dataset of Whole Slide Images (WSI) for Multiple Instance Learning (MIL) that was processed using the [TRIDENT](https://github.com/mahmoodlab/TRIDENT) repository.

    **Directory structure.**
    For more information on the processing of the bags, refer to the [`ProcessedMILDataset` class](processed_mil_dataset.md).
    This dataset expects the directory structure provided by the TRIDENT repository. Given the base path, a magnification `mag`, a patch size `ps`, and a number of overlapping pixels `opx`, TRIDENT will create (among others) the folder `{mag}x_{ps}px_{opx}px_overlap/`. In this folder, the following folders is expected:
    ```
    features_{feature_extractor}
    ├── wsi1.h5
    ├── wsi2.h5
    └── ...
    patches
    ├── wsi1.h5
    ├── wsi2.h5
    └── ...
    ```

    **Adjacency matrix.**
    If the coordinates of the patches are available, an adjacency matrix representing the spatial relationships between the patches is built. Please refer to the [`ProcessedMILDataset` class](processed_mil_dataset.md) for more information on how the adjacency matrix is built.

    **Labels loading.**
    The labels of the WSIs can be provided in two ways:
    1. As a directory containing one file per WSI, following the same structure as the features and patches folders.
    2. As a CSV file containing the WSI names and their corresponding labels. In this case, the user must provide the column names for the WSI names and labels using the `wsi_name_col` and `wsi_label_col` keyword arguments, respectively.

    """

    def __init__(
        self,
        base_path: str,
        labels_path: str,
        feature_extractor: str = "conch_v15",
        magnification: int = 20,
        patch_size: int = 512,
        overlap_pixels: int = 0,
        patch_labels_path: str = None,
        wsi_names: list = None,
        bag_keys: list = ["X", "Y", "y_inst", "adj", "coords"],
        dist_thr: float = None,
        adj_with_dist: bool = False,
        norm_adj: bool = True,
        load_at_init: bool = True,
        **kwargs,
    ) -> None:
        """
        Class constructor.

        Arguments:
            base_path: Path to the base directory containing the TRIDENT folders.
            labels_path: Path to the directory or CSV file containing the labels of the WSIs.
            feature_extractor: Feature extractor used to extract the features. This will determine the features folder name.
            magnification: Magnification used to extract the patches.
            patch_size: Size of the patches.
            overlap_pixels: Number of overlapping pixels between patches.
            patch_labels_path: Path to the directory containing the instance-level labels of the patches.
            wsi_names: List of WSI names to include in the dataset.
            bag_keys: List of keys to use for the bags. Must be in ['X', 'Y', 'y_inst', 'adj', 'coords'].
            dist_thr: Distance threshold for building the adjacency matrix.
            adj_with_dist: If True, the adjacency matrix is built using the Euclidean distance between the patches features. If False, the adjacency matrix is binary.
            norm_adj: If True, normalize the adjacency matrix.
            load_at_init: If True, load the bags at initialization. If False, load the bags on demand.
            kwargs (Any): Additional keyword arguments. Used for passing column names when labels_path is a CSV file.

        """
        if dist_thr is None:
            # dist_thr = np.sqrt(2.0) * patch_size
            dist_thr = np.sqrt(2.0)
        self.patch_size = patch_size
        self.feature_extractor = feature_extractor
        self.magnification = magnification
        self.overlap_pixels = overlap_pixels
        self.trident_folder = (
            f"{magnification}x_{patch_size}px_{overlap_pixels}px_overlap/"
        )
        self.kwargs = kwargs

        super().__init__(
            features_path=base_path
            + self.trident_folder
            + f"features_{feature_extractor}/",
            labels_path=labels_path,
            patch_labels_path=patch_labels_path,
            coords_path=base_path + self.trident_folder + "patches/",
            wsi_names=wsi_names,
            bag_keys=bag_keys,
            file_type=".h5",
            dist_thr=dist_thr,
            adj_with_dist=adj_with_dist,
            norm_adj=norm_adj,
            load_at_init=load_at_init,
        )

    def _load_labels(self, name: str) -> np.ndarray:
        """
        Load the labels of a bag from disk. This function adds the functionality of reading the bag labels from a CSV, checking if the provided path is a directory or a file.
        To achieve this, it is assumed that the

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
            if "wsi_name_col" in self.kwargs and "wsi_label_col" in self.kwargs:
                wsi_name_col = self.kwargs["wsi_name_col"]
                wsi_label_col = self.kwargs["wsi_label_col"]
                try:
                    labels = self.labels_csv.loc[
                        self.labels_csv[wsi_name_col] == name, wsi_label_col
                    ].values
                except ValueError:
                    raise ValueError(
                        f"Could not read the label of the file {name} from the CSV file {self.labels_path}. Please check that the column names provided in 'wsi_name_col' and 'wsi_label_col' are correct."
                    )
            else:
                raise ValueError(
                    "When providing a CSV file for labels_path, you must provide 'wsi_name_col' and 'wsi_label_col' in the kwargs."
                )
            return labels

    def _load_coords(self, name: str) -> np.ndarray:
        """
        Load the coordinates of a bag from disk.

        Arguments:
            name: Name of the bag to load.

        Returns:
            coords: Coordinates of the bag.
        """
        coords_file = os.path.join(self.coords_path, name + "_patches" + self.file_type)
        coords = h5py.File(coords_file, "r")["coords"][:]
        if coords is not None:
            coords = coords / self.patch_size
            min_coords = np.min(coords, axis=0)
            coords = coords - min_coords
            coords = coords.astype(int)
        return coords
