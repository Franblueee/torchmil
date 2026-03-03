import pytest
import numpy as np
import pandas as pd
import h5py
import torch
from tensordict import TensorDict

from torchmil.datasets import TridentWSIDataset


# --- Fixtures for creating temporary data ---


@pytest.fixture(scope="function")
def temp_data_dir(tmp_path):
    """
    Creates a temporary directory for dataset files and returns its path.
    """
    data_dir = tmp_path / "test_trident_data"
    data_dir.mkdir()
    yield data_dir


@pytest.fixture(scope="function")
def create_dummy_h5_files(temp_data_dir):
    """
    Creates a set of dummy .h5 files for a specified WSI name following TRIDENT structure.
    """

    def _create_files(
        wsi_name,
        base_path,
        feature_extractor="UNI",
        bag_size=5,
        feature_dim=128,
        coord_dim=2,
        patch_size=512,
        create_features=True,
        create_coords=True,
        create_patch_labels=False,
    ):
        # Create features if requested
        if create_features:
            features_path = base_path / f"features_{feature_extractor}"
            features_path.mkdir(parents=True, exist_ok=True)
            with h5py.File(features_path / f"{wsi_name}.h5", "w") as f:
                f.create_dataset(
                    "features",
                    data=np.random.rand(bag_size, feature_dim).astype(np.float32),
                )

        # Create coords if requested (in patches folder with _patches suffix)
        if create_coords:
            coords_path = base_path / "patches"
            coords_path.mkdir(parents=True, exist_ok=True)
            coords = (np.random.rand(bag_size, coord_dim) * patch_size * 10).astype(
                np.float32
            )
            with h5py.File(coords_path / f"{wsi_name}_patches.h5", "w") as f:
                f.create_dataset("coords", data=coords)

        # Create patch labels if requested
        if create_patch_labels:
            patch_labels_path = base_path / "patch_labels"
            patch_labels_path.mkdir(parents=True, exist_ok=True)
            with h5py.File(patch_labels_path / f"{wsi_name}.h5", "w") as f:
                f.create_dataset(
                    "patch_labels",
                    data=np.random.randint(0, 2, size=bag_size).astype(np.float32),
                )

    return _create_files


@pytest.fixture(scope="function")
def create_labels_dir(temp_data_dir):
    """
    Creates a labels directory with .h5 label files.
    """

    def _create_labels(wsi_names, base_path):
        labels_path = base_path / "labels"
        labels_path.mkdir(parents=True, exist_ok=True)
        for wsi_name in wsi_names:
            with h5py.File(labels_path / f"{wsi_name}.h5", "w") as f:
                f.create_dataset(
                    "labels",
                    data=np.array([np.random.randint(0, 2)]).astype(np.float32),
                )
        return str(labels_path)

    return _create_labels


@pytest.fixture(scope="function")
def create_labels_csv(temp_data_dir):
    """
    Creates a CSV file containing WSI names and labels.
    """

    def _create_csv(
        wsi_names, base_path, wsi_name_col="wsi_name", wsi_label_col="label"
    ):
        csv_path = base_path / "labels.csv"
        labels = np.random.randint(0, 2, size=len(wsi_names))
        df = pd.DataFrame({wsi_name_col: wsi_names, wsi_label_col: labels})
        df.to_csv(csv_path, index=False)
        return str(csv_path), wsi_name_col, wsi_label_col

    return _create_csv


@pytest.fixture(scope="function")
def setup_full_trident_dataset(temp_data_dir, create_dummy_h5_files, create_labels_dir):
    """
    Sets up a complete TRIDENT dataset with features, coords, and labels.
    """
    base_path = temp_data_dir / "trident_output" / "20x_512px_0px_overlap"
    base_path.mkdir(parents=True, exist_ok=True)

    wsi_names = ["wsi_0", "wsi_1", "wsi_2"]
    feature_extractor = "UNI"

    for name in wsi_names:
        create_dummy_h5_files(
            name,
            base_path,
            feature_extractor=feature_extractor,
            create_features=True,
            create_coords=True,
            create_patch_labels=True,
        )

    labels_path = create_labels_dir(wsi_names, base_path)

    return {
        "base_path": str(base_path) + "/",
        "labels_path": labels_path,
        "feature_extractor": feature_extractor,
        "patch_labels_path": str(base_path / "patch_labels"),
        "wsi_names": wsi_names,
    }


@pytest.fixture(scope="function")
def setup_trident_dataset_with_csv_labels(
    temp_data_dir, create_dummy_h5_files, create_labels_csv
):
    """
    Sets up a TRIDENT dataset with labels provided via CSV file.
    """
    base_path = temp_data_dir / "trident_csv" / "20x_512px_0px_overlap"
    base_path.mkdir(parents=True, exist_ok=True)

    wsi_names = ["wsi_a", "wsi_b", "wsi_c"]
    feature_extractor = "CONCH"

    for name in wsi_names:
        create_dummy_h5_files(
            name,
            base_path,
            feature_extractor=feature_extractor,
            create_features=True,
            create_coords=True,
            create_patch_labels=False,
        )

    labels_path, wsi_name_col, wsi_label_col = create_labels_csv(wsi_names, base_path)

    return {
        "base_path": str(base_path) + "/",
        "labels_path": labels_path,
        "feature_extractor": feature_extractor,
        "wsi_names": wsi_names,
        "wsi_name_col": wsi_name_col,
        "wsi_label_col": wsi_label_col,
    }


# --- Test Cases ---


class TestTridentWSIDatasetInitialization:
    """Tests for dataset initialization."""

    def test_initialization_with_full_data(self, setup_full_trident_dataset):
        """Tests successful initialization with all data paths provided."""
        dataset = TridentWSIDataset(
            base_path=setup_full_trident_dataset["base_path"],
            labels_path=setup_full_trident_dataset["labels_path"],
            feature_extractor=setup_full_trident_dataset["feature_extractor"],
            patch_labels_path=setup_full_trident_dataset["patch_labels_path"],
            bag_keys=["X", "Y", "y_inst", "adj", "coords"],
            load_at_init=True,
        )
        assert len(dataset) == len(setup_full_trident_dataset["wsi_names"])
        assert len(dataset.loaded_bags) == len(setup_full_trident_dataset["wsi_names"])

    def test_initialization_with_features_and_labels_only(
        self, setup_full_trident_dataset
    ):
        """Tests initialization with only features and labels."""
        dataset = TridentWSIDataset(
            base_path=setup_full_trident_dataset["base_path"],
            labels_path=setup_full_trident_dataset["labels_path"],
            feature_extractor=setup_full_trident_dataset["feature_extractor"],
            bag_keys=["X", "Y"],
            load_at_init=True,
        )
        assert len(dataset) == len(setup_full_trident_dataset["wsi_names"])

        bag = dataset[0]
        assert "X" in bag
        assert "Y" in bag
        assert "y_inst" not in bag
        assert "adj" not in bag
        assert "coords" not in bag

    def test_initialization_with_csv_labels(
        self, setup_trident_dataset_with_csv_labels
    ):
        """Tests initialization with labels from CSV file."""
        dataset = TridentWSIDataset(
            base_path=setup_trident_dataset_with_csv_labels["base_path"],
            labels_path=setup_trident_dataset_with_csv_labels["labels_path"],
            feature_extractor=setup_trident_dataset_with_csv_labels[
                "feature_extractor"
            ],
            bag_keys=["X", "Y"],
            wsi_name_col=setup_trident_dataset_with_csv_labels["wsi_name_col"],
            wsi_label_col=setup_trident_dataset_with_csv_labels["wsi_label_col"],
            load_at_init=True,
        )
        assert len(dataset) == len(setup_trident_dataset_with_csv_labels["wsi_names"])

    def test_initialization_with_specific_wsi_names(self, setup_full_trident_dataset):
        """Tests initialization with specific WSI names."""
        wsi_subset = ["wsi_0", "wsi_2"]
        dataset = TridentWSIDataset(
            base_path=setup_full_trident_dataset["base_path"],
            labels_path=setup_full_trident_dataset["labels_path"],
            feature_extractor=setup_full_trident_dataset["feature_extractor"],
            wsi_names=wsi_subset,
            bag_keys=["X", "Y"],
            load_at_init=True,
        )
        assert len(dataset) == len(wsi_subset)

    def test_default_dist_thr_calculation(self, setup_full_trident_dataset):
        """Tests that dist_thr defaults to sqrt(2) when not provided."""
        dataset = TridentWSIDataset(
            base_path=setup_full_trident_dataset["base_path"],
            labels_path=setup_full_trident_dataset["labels_path"],
            feature_extractor=setup_full_trident_dataset["feature_extractor"],
            bag_keys=["X", "Y"],
            load_at_init=False,
        )
        assert dataset.dist_thr == pytest.approx(np.sqrt(2.0))

    def test_custom_dist_thr(self, setup_full_trident_dataset):
        """Tests initialization with custom dist_thr."""
        custom_dist_thr = 3.0
        dataset = TridentWSIDataset(
            base_path=setup_full_trident_dataset["base_path"],
            labels_path=setup_full_trident_dataset["labels_path"],
            feature_extractor=setup_full_trident_dataset["feature_extractor"],
            bag_keys=["X", "Y"],
            dist_thr=custom_dist_thr,
            load_at_init=False,
        )
        assert dataset.dist_thr == custom_dist_thr


class TestTridentWSIDatasetLoadMethods:
    """Tests for data loading methods."""

    def test_load_features(self, setup_full_trident_dataset):
        """Tests that features are loaded correctly."""
        dataset = TridentWSIDataset(
            base_path=setup_full_trident_dataset["base_path"],
            labels_path=setup_full_trident_dataset["labels_path"],
            feature_extractor=setup_full_trident_dataset["feature_extractor"],
            bag_keys=["X", "Y"],
            load_at_init=True,
        )
        bag = dataset[0]
        assert "X" in bag
        assert isinstance(bag["X"], torch.Tensor)
        assert bag["X"].shape[1] == 128  # Feature dimension

    def test_load_coords(self, setup_full_trident_dataset):
        """Tests that coordinates are loaded and normalized by patch_size."""
        patch_size = 512
        dataset = TridentWSIDataset(
            base_path=setup_full_trident_dataset["base_path"],
            labels_path=setup_full_trident_dataset["labels_path"],
            feature_extractor=setup_full_trident_dataset["feature_extractor"],
            bag_keys=["X", "Y", "coords"],
            patch_size=patch_size,
            load_at_init=True,
        )
        bag = dataset[0]
        assert "coords" in bag
        assert isinstance(bag["coords"], torch.Tensor)
        # Coordinates should be normalized (divided by patch_size and converted to int)
        assert bag["coords"].dtype == torch.int64 or bag["coords"].dtype == torch.int32

    def test_load_inst_labels(self, setup_full_trident_dataset):
        """Tests that instance labels are loaded correctly."""
        dataset = TridentWSIDataset(
            base_path=setup_full_trident_dataset["base_path"],
            labels_path=setup_full_trident_dataset["labels_path"],
            feature_extractor=setup_full_trident_dataset["feature_extractor"],
            patch_labels_path=setup_full_trident_dataset["patch_labels_path"],
            bag_keys=["X", "Y", "y_inst"],
            load_at_init=True,
        )
        bag = dataset[0]
        assert "y_inst" in bag
        assert isinstance(bag["y_inst"], torch.Tensor)
        assert bag["y_inst"].shape[0] == bag["X"].shape[0]

    def test_load_labels_from_directory(self, setup_full_trident_dataset):
        """Tests that labels are loaded correctly from a directory."""
        dataset = TridentWSIDataset(
            base_path=setup_full_trident_dataset["base_path"],
            labels_path=setup_full_trident_dataset["labels_path"],
            feature_extractor=setup_full_trident_dataset["feature_extractor"],
            bag_keys=["X", "Y"],
            load_at_init=True,
        )
        bag = dataset[0]
        assert "Y" in bag
        assert isinstance(bag["Y"], torch.Tensor)

    def test_load_labels_from_csv(self, setup_trident_dataset_with_csv_labels):
        """Tests that labels are loaded correctly from a CSV file."""
        dataset = TridentWSIDataset(
            base_path=setup_trident_dataset_with_csv_labels["base_path"],
            labels_path=setup_trident_dataset_with_csv_labels["labels_path"],
            feature_extractor=setup_trident_dataset_with_csv_labels[
                "feature_extractor"
            ],
            bag_keys=["X", "Y"],
            wsi_name_col=setup_trident_dataset_with_csv_labels["wsi_name_col"],
            wsi_label_col=setup_trident_dataset_with_csv_labels["wsi_label_col"],
            load_at_init=True,
        )
        bag = dataset[0]
        assert "Y" in bag
        assert isinstance(bag["Y"], torch.Tensor)

    def test_load_adjacency_matrix(self, setup_full_trident_dataset):
        """Tests that adjacency matrix is built correctly."""
        dataset = TridentWSIDataset(
            base_path=setup_full_trident_dataset["base_path"],
            labels_path=setup_full_trident_dataset["labels_path"],
            feature_extractor=setup_full_trident_dataset["feature_extractor"],
            bag_keys=["X", "Y", "adj"],
            load_at_init=True,
        )
        bag = dataset[0]
        assert "adj" in bag
        assert isinstance(bag["adj"], torch.Tensor)
        assert bag["adj"].is_sparse
        assert bag["adj"].shape[0] == bag["X"].shape[0]


class TestTridentWSIDatasetCSVLabels:
    """Tests for CSV label loading functionality."""

    def test_csv_labels_missing_column_names(
        self, setup_trident_dataset_with_csv_labels
    ):
        """Tests error when CSV columns are not specified."""
        with pytest.raises(ValueError, match="wsi_name_col.*wsi_label_col"):
            _ = TridentWSIDataset(
                base_path=setup_trident_dataset_with_csv_labels["base_path"],
                labels_path=setup_trident_dataset_with_csv_labels["labels_path"],
                feature_extractor=setup_trident_dataset_with_csv_labels[
                    "feature_extractor"
                ],
                bag_keys=["X", "Y"],
                # Missing wsi_name_col and wsi_label_col
                load_at_init=True,
            )

    def test_csv_labels_with_file_extension_in_names(
        self, temp_data_dir, create_dummy_h5_files
    ):
        """Tests CSV label loading when WSI names have file extensions."""
        base_path = temp_data_dir / "trident_ext_test" / "20x_512px_0px_overlap"
        base_path.mkdir(parents=True, exist_ok=True)

        wsi_names = ["wsi_1", "wsi_2"]
        feature_extractor = "UNI"

        for name in wsi_names:
            create_dummy_h5_files(
                name,
                base_path,
                feature_extractor=feature_extractor,
                create_features=True,
                create_coords=True,
            )

        # Create CSV with .svs extension in names
        csv_path = base_path / "labels.csv"
        df = pd.DataFrame(
            {
                "slide_id": [f"{name}.svs" for name in wsi_names],
                "diagnosis": [0, 1],
            }
        )
        df.to_csv(csv_path, index=False)

        dataset = TridentWSIDataset(
            base_path=str(base_path) + "/",
            labels_path=str(csv_path),
            feature_extractor=feature_extractor,
            bag_keys=["X", "Y"],
            wsi_name_col="slide_id",
            wsi_label_col="diagnosis",
            load_at_init=True,
        )

        # Should successfully load labels despite extension in CSV
        bag = dataset[0]
        assert "Y" in bag


class TestTridentWSIDatasetErrorHandling:
    """Tests for error handling."""

    def test_missing_features_directory(self, temp_data_dir, create_labels_dir):
        """Tests error when features directory doesn't exist."""
        base_path = temp_data_dir / "empty_trident"
        base_path.mkdir(parents=True, exist_ok=True)
        labels_path = create_labels_dir(["wsi_test"], base_path)

        with pytest.raises(FileNotFoundError):
            TridentWSIDataset(
                base_path=str(base_path) + "/",
                labels_path=labels_path,
                feature_extractor="NONEXISTENT",
                bag_keys=["X", "Y"],
                load_at_init=False,
            )

    def test_missing_specific_feature_file(self, setup_full_trident_dataset):
        """Tests error when a specific feature file is missing."""
        dataset = TridentWSIDataset(
            base_path=setup_full_trident_dataset["base_path"],
            labels_path=setup_full_trident_dataset["labels_path"],
            feature_extractor=setup_full_trident_dataset["feature_extractor"],
            wsi_names=[
                "wsi_0",
                "zzz_nonexistent_wsi",
            ],  # zzz prefix ensures it's sorted last
            bag_keys=["X", "Y"],
            load_at_init=False,
        )

        # First bag (wsi_0) should load successfully
        _ = dataset[0]

        # Second bag (nonexistent) should raise error
        with pytest.raises((FileNotFoundError, ValueError)):
            _ = dataset[1]


class TestTridentWSIDatasetGetterMethods:
    """Tests for getter methods."""

    def test_len_method(self, setup_full_trident_dataset):
        """Tests the __len__ method."""
        dataset = TridentWSIDataset(
            base_path=setup_full_trident_dataset["base_path"],
            labels_path=setup_full_trident_dataset["labels_path"],
            feature_extractor=setup_full_trident_dataset["feature_extractor"],
            bag_keys=["X", "Y"],
            load_at_init=True,
        )
        assert len(dataset) == len(setup_full_trident_dataset["wsi_names"])

    def test_get_bag_names_method(self, setup_full_trident_dataset):
        """Tests the get_bag_names method."""
        dataset = TridentWSIDataset(
            base_path=setup_full_trident_dataset["base_path"],
            labels_path=setup_full_trident_dataset["labels_path"],
            feature_extractor=setup_full_trident_dataset["feature_extractor"],
            bag_keys=["X", "Y"],
            load_at_init=True,
        )
        bag_names = dataset.get_bag_names()
        assert isinstance(bag_names, list)
        assert len(bag_names) == len(setup_full_trident_dataset["wsi_names"])
        # Names should be sorted
        assert bag_names == sorted(bag_names)

    def test_getitem_returns_tensordict(self, setup_full_trident_dataset):
        """Tests that __getitem__ returns a TensorDict."""
        dataset = TridentWSIDataset(
            base_path=setup_full_trident_dataset["base_path"],
            labels_path=setup_full_trident_dataset["labels_path"],
            feature_extractor=setup_full_trident_dataset["feature_extractor"],
            bag_keys=["X", "Y"],
            load_at_init=True,
        )
        bag = dataset[0]
        assert isinstance(bag, TensorDict)


class TestTridentWSIDatasetSubset:
    """Tests for subset functionality."""

    def test_subset_method(self, setup_full_trident_dataset):
        """Tests the subset method."""
        dataset = TridentWSIDataset(
            base_path=setup_full_trident_dataset["base_path"],
            labels_path=setup_full_trident_dataset["labels_path"],
            feature_extractor=setup_full_trident_dataset["feature_extractor"],
            bag_keys=["X", "Y"],
            load_at_init=True,
        )

        original_bag_names = dataset.get_bag_names()
        subset_indices = [0, 2]
        subset_dataset = dataset.subset(subset_indices)

        assert len(subset_dataset) == len(subset_indices)
        assert subset_dataset.get_bag_names() == [
            original_bag_names[i] for i in subset_indices
        ]


class TestTridentWSIDatasetAdjacencyOptions:
    """Tests for adjacency matrix options."""

    def test_adj_with_dist(self, setup_full_trident_dataset):
        """Tests adjacency matrix with distance weighting."""
        dataset = TridentWSIDataset(
            base_path=setup_full_trident_dataset["base_path"],
            labels_path=setup_full_trident_dataset["labels_path"],
            feature_extractor=setup_full_trident_dataset["feature_extractor"],
            bag_keys=["X", "Y", "adj"],
            adj_with_dist=True,
            load_at_init=True,
        )
        bag = dataset[0]
        assert "adj" in bag
        assert bag["adj"].is_sparse

    def test_normalized_adj(self, setup_full_trident_dataset):
        """Tests normalized adjacency matrix."""
        dataset = TridentWSIDataset(
            base_path=setup_full_trident_dataset["base_path"],
            labels_path=setup_full_trident_dataset["labels_path"],
            feature_extractor=setup_full_trident_dataset["feature_extractor"],
            bag_keys=["X", "Y", "adj"],
            norm_adj=True,
            load_at_init=True,
        )
        bag = dataset[0]
        assert "adj" in bag

    def test_unnormalized_adj(self, setup_full_trident_dataset):
        """Tests unnormalized adjacency matrix."""
        dataset = TridentWSIDataset(
            base_path=setup_full_trident_dataset["base_path"],
            labels_path=setup_full_trident_dataset["labels_path"],
            feature_extractor=setup_full_trident_dataset["feature_extractor"],
            bag_keys=["X", "Y", "adj"],
            norm_adj=False,
            load_at_init=True,
        )
        bag = dataset[0]
        assert "adj" in bag


class TestTridentWSIDatasetLazyLoading:
    """Tests for lazy loading functionality."""

    def test_lazy_loading(self, setup_full_trident_dataset):
        """Tests that bags are not loaded at init when load_at_init=False."""
        dataset = TridentWSIDataset(
            base_path=setup_full_trident_dataset["base_path"],
            labels_path=setup_full_trident_dataset["labels_path"],
            feature_extractor=setup_full_trident_dataset["feature_extractor"],
            bag_keys=["X", "Y"],
            load_at_init=False,
        )
        assert len(dataset.loaded_bags) == 0

        # Access a bag to trigger loading
        _ = dataset[0]
        assert len(dataset.loaded_bags) == 1

    def test_eager_loading(self, setup_full_trident_dataset):
        """Tests that all bags are loaded at init when load_at_init=True."""
        dataset = TridentWSIDataset(
            base_path=setup_full_trident_dataset["base_path"],
            labels_path=setup_full_trident_dataset["labels_path"],
            feature_extractor=setup_full_trident_dataset["feature_extractor"],
            bag_keys=["X", "Y"],
            load_at_init=True,
        )
        assert len(dataset.loaded_bags) == len(setup_full_trident_dataset["wsi_names"])
