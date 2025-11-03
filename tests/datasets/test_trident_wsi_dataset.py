import pytest
import numpy as np
import os
import h5py
from pathlib import Path
from torchmil.datasets import TridentWSIDataset  # Update to actual import path


# --- Helper Functions for Fixtures (Kept for creating H5 files) ---


def create_h5_file(filepath: Path, dataset_name: str, data: np.ndarray):
    """Helper to create a simple HDF5 file with one dataset."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(filepath, "w") as f:
        f.create_dataset(dataset_name, data=data)


# --- Pytest Fixtures ---


@pytest.fixture
def mock_trident_data(tmp_path):
    """
    Sets up a minimal TRIDENT directory structure using H5 files.
    """
    WSI_NAME = "sample"
    MAG, PS, OPX = 20, 512, 0
    FEAT_EXT = "conch_v15"
    TRIDENT_FOLDER = f"{MAG}x_{PS}px_{OPX}px_overlap"

    base_path = tmp_path / "trident_base"
    full_trident_path = base_path / TRIDENT_FOLDER

    features_dir = full_trident_path / f"features_{FEAT_EXT}"
    labels_dir = full_trident_path / "labels"
    coords_dir = full_trident_path / "patches"
    inst_labels_dir = full_trident_path / "patch_labels"

    # Create directories
    for d in [features_dir, labels_dir, coords_dir, inst_labels_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Raw coordinates (must be divisible by PS=512 for clean test)
    # Scaled and normalized result: [[2, 4], [3, 5], [1, 2]] - [1, 2] = [[1, 2], [2, 3], [0, 0]]
    raw_coords = np.array([[1024, 2048], [1536, 2560], [512, 1024]]).astype(np.int32)

    # 1. Features file
    create_h5_file(
        features_dir / f"{WSI_NAME}.h5",
        "features",
        np.random.rand(raw_coords.shape[0], 128),
    )

    # 2. Label file (WSI-level)
    create_h5_file(
        labels_dir / f"{WSI_NAME}.h5", "label", np.array([1], dtype=np.float32)
    )

    # 3. Coords file
    create_h5_file(coords_dir / f"{WSI_NAME}_patches.h5", "coords", raw_coords)

    # 4. Patch Label file (Instance-level)
    create_h5_file(
        inst_labels_dir / f"{WSI_NAME}.h5",
        "patch_label",
        np.random.randint(0, 2, size=(raw_coords.shape[0],)),
    )

    return {
        "base_path": str(base_path) + os.sep,
        "labels_path": str(labels_dir) + os.sep,
        "patch_labels_path": str(inst_labels_dir) + os.sep,
        "wsi_names": [WSI_NAME],
        "patch_size": PS,
        # The other params are defaults, but included for clarity in TridentWSIDataset
        "feature_extractor": FEAT_EXT,
        "magnification": MAG,
        "overlap_pixels": OPX,
        "adj_with_dist": False,
        "norm_adj": True,
        "load_at_init": False,
        "expected_coords": np.array([[1, 2], [2, 3], [0, 0]]).astype(np.int32),
    }


def test_trident_dataset_init(mock_trident_data):
    """
    Tests initialization of TridentWSIDataset to check if core attributes are set correctly.
    """
    # Assuming TridentWSIDataset is imported or defined above
    dataset = TridentWSIDataset(**mock_trident_data)

    # Check attributes set by TridentWSIDataset's __init__
    assert dataset.patch_size == 512
    assert "conch_v15" in dataset.feature_extractor

    # Check attributes passed to super() and derived paths
    assert (
        "trident_base/20x_512px_0px_overlap/features_conch_v15/"
        in dataset.features_path
    )
    assert "trident_base/20x_512px_0px_overlap/patches/" in dataset.coords_path
    assert dataset.bag_names == ["sample"]


def test_trident_load_coords_adjustment(mock_trident_data):
    """
    Tests the overridden _load_coords method for correct scaling and normalization.
    """
    # Assuming TridentWSIDataset is imported or defined above
    dataset = TridentWSIDataset(**mock_trident_data)
    bag_name = mock_trident_data["wsi_names"][0]

    # Check for the presence of the method we expect to be defined
    assert hasattr(dataset, "_load_coords")

    loaded_coords = dataset._load_coords(bag_name)

    assert loaded_coords is not None
    assert isinstance(loaded_coords, np.ndarray)
    print(loaded_coords.dtype)

    # Check normalization and casting as per Trident's logic
    assert loaded_coords.min() == 0, "Coordinates were not normalized (min-subtracted)."
    assert loaded_coords.dtype == np.int_, "Coordinates should be cast to integer."
    assert np.array_equal(
        loaded_coords, mock_trident_data["expected_coords"]
    ), "Coordinate calculation is incorrect."
