import pytest
import numpy as np
import torch
from tensordict import TensorDict
from pathlib import Path

from torchmil.datasets import ProcessedMILDataset

# --- Fixtures for creating temporary data ---


@pytest.fixture(scope="function")
def temp_data_dir(tmp_path):
    """
    Creates a temporary directory for dataset files and returns its path.
    Ensures a clean slate for each test function.
    """
    data_dir = tmp_path / "test_mil_data"
    data_dir.mkdir()
    yield data_dir
    # Cleanup is handled automatically by tmp_path fixture


@pytest.fixture(scope="function")
def create_dummy_bag_files(temp_data_dir):
    """
    Creates a set of dummy .npy files for a specified bag name.
    """

    def _create_files(
        bag_name,
        features_path=None,
        labels_path=None,
        inst_labels_path=None,
        coords_path=None,
        bag_size=5,
        feature_dim=128,
        coord_dim=2,
    ):
        # Create features if path is provided
        if features_path:
            features_path.mkdir(parents=True, exist_ok=True)
            np.save(
                features_path / f"{bag_name}.npy",
                np.random.rand(bag_size, feature_dim).astype(np.float32),
            )

        # Create labels if path is provided
        if labels_path:
            labels_path.mkdir(parents=True, exist_ok=True)
            np.save(
                labels_path / f"{bag_name}.npy",
                np.array([np.random.randint(0, 2)]).astype(np.float32),
            )

        # Create optional files if paths are provided
        if inst_labels_path:
            inst_labels_path.mkdir(parents=True, exist_ok=True)
            np.save(
                inst_labels_path / f"{bag_name}.npy",
                np.random.randint(0, 2, size=bag_size).astype(np.float32),
            )
        if coords_path:
            coords_path.mkdir(parents=True, exist_ok=True)
            np.save(
                coords_path / f"{bag_name}.npy",
                np.random.rand(bag_size, coord_dim).astype(np.float32),
            )

    return _create_files


@pytest.fixture(scope="function")
def setup_full_dataset_paths(temp_data_dir, create_dummy_bag_files):
    """
    Sets up a complete set of dummy files for a dataset with multiple bags.
    Returns paths to the data directories.
    """
    features_dir = temp_data_dir / "features"
    labels_dir = temp_data_dir / "labels"
    inst_labels_dir = temp_data_dir / "inst_labels"
    coords_dir = temp_data_dir / "coords"

    bag_names = ["bag_0", "bag_1", "bag_2"]

    for name in bag_names:
        create_dummy_bag_files(
            name, features_dir, labels_dir, inst_labels_dir, coords_dir
        )

    return {
        "features_path": str(features_dir),
        "labels_path": str(labels_dir),
        "inst_labels_path": str(inst_labels_dir),
        "coords_path": str(coords_dir),
        "bag_names": bag_names,
    }


# --- Test Cases ---


def test_dataset_initialization_full_data(setup_full_dataset_paths):
    """
    Tests successful initialization with all data paths provided and all keys requested.
    """
    dataset = ProcessedMILDataset(
        features_path=setup_full_dataset_paths["features_path"],
        labels_path=setup_full_dataset_paths["labels_path"],
        inst_labels_path=setup_full_dataset_paths["inst_labels_path"],
        coords_path=setup_full_dataset_paths["coords_path"],
        bag_keys=["X", "Y", "y_inst", "adj", "coords"],
        load_at_init=True,
    )
    assert len(dataset) == len(setup_full_dataset_paths["bag_names"])
    assert len(dataset.loaded_bags) == len(setup_full_dataset_paths["bag_names"])

    # Check a specific bag
    bag_0 = dataset[0]
    assert "X" in bag_0
    assert "Y" in bag_0
    assert "y_inst" in bag_0
    assert "coords" in bag_0
    assert "adj" in bag_0  # Adj should be built if coords are present and requested
    assert isinstance(bag_0["X"], torch.Tensor)
    assert bag_0["X"].shape[0] > 0  # Ensure instances are loaded


def test_dataset_initialization_features_labels_only_explicit_keys(
    temp_data_dir, create_dummy_bag_files
):
    """
    Tests successful initialization with only mandatory features and labels, and explicit keys.
    """
    features_dir = temp_data_dir / "features_only"
    labels_dir = temp_data_dir / "labels_only"
    bag_names = ["bag_A", "bag_B"]

    for name in bag_names:
        create_dummy_bag_files(
            name, features_path=features_dir, labels_path=labels_dir
        )  # No inst_labels or coords

    dataset = ProcessedMILDataset(
        features_path=str(features_dir),
        labels_path=str(labels_dir),
        bag_keys=["X", "Y"],  # Only request mandatory keys
        load_at_init=True,
    )
    assert len(dataset) == len(bag_names)
    bag_A = dataset[0]
    assert "X" in bag_A
    assert "Y" in bag_A
    assert (
        "y_inst" not in bag_A
    )  # Should not be present if path not provided and not in bag_keys
    assert (
        "coords" not in bag_A
    )  # Should not be present if path not provided and not in bag_keys
    assert (
        "adj" not in bag_A
    )  # Should not be present if coords not present and not in bag_keys


# # --- New ValueError Scenarios in __init__ ---


def test_value_error_x_in_bag_keys_but_features_path_none():
    """
    Tests ValueError if 'X' is in bag_keys but features_path is None.
    """
    with pytest.raises(ValueError):
        ProcessedMILDataset(
            features_path=None,  # Explicitly None
            labels_path="dummy_path",  # Needs a path for bag_names discovery, but will fail earlier
            bag_keys=["X", "Y"],
        )


def test_value_error_y_in_bag_keys_but_labels_path_none():
    """
    Tests ValueError if 'Y' is in bag_keys but labels_path is None.
    """
    with pytest.raises(ValueError):
        ProcessedMILDataset(
            features_path="dummy_path",
            labels_path=None,  # Explicitly None
            bag_keys=["X", "Y"],
        )


def test_value_error_y_inst_in_bag_keys_but_inst_labels_path_none():
    """
    Tests ValueError if 'y_inst' is in bag_keys but inst_labels_path is None.
    """
    with pytest.raises(ValueError):
        ProcessedMILDataset(
            features_path="dummy_path",
            labels_path="dummy_path",
            inst_labels_path=None,  # Explicitly None
            bag_keys=["X", "Y", "y_inst"],
        )


def test_value_error_coords_in_bag_keys_but_coords_path_none():
    """
    Tests ValueError if 'coords' is in bag_keys but coords_path is None.
    """
    with pytest.raises(ValueError):
        ProcessedMILDataset(
            features_path="dummy_path",
            labels_path="dummy_path",
            coords_path=None,  # Explicitly None
            bag_keys=["X", "Y", "coords"],
        )


def test_value_error_adj_in_bag_keys_but_coords_path_none():
    """
    Tests ValueError if 'adj' is in bag_keys but coords_path is None.
    """
    with pytest.raises(ValueError):
        ProcessedMILDataset(
            features_path="dummy_path",
            labels_path="dummy_path",
            coords_path=None,  # Explicitly None
            bag_keys=["X", "Y", "adj"],
        )


# # --- Fail Fast Scenarios for Missing Files (after path validation) ---


def test_fail_fast_missing_features_dir_during_bag_names_discovery(temp_data_dir):
    """
    Tests FileNotFoundError when features_path is a non-existent directory
    and bag_names is None (triggering os.listdir).
    """
    non_existent_path = str(temp_data_dir / "non_existent_features_dir")
    with pytest.raises(FileNotFoundError):
        # bag_names is None, so it tries to listdir features_path
        ProcessedMILDataset(
            features_path=non_existent_path,
            labels_path="dummy_labels_path",  # Needs to be non-None to pass initial ValueError for Y
            bag_keys=["X", "Y"],
            load_at_init=False,
        )


def test_fail_fast_missing_feature_file(
    setup_full_dataset_paths, create_dummy_bag_files, tmp_path
):
    """
    Tests FileNotFoundError when a specific feature file is missing.
    """
    features_dir = Path(setup_full_dataset_paths["features_path"])
    labels_dir = Path(setup_full_dataset_paths["labels_path"])

    # Create dummy files for two bags
    bag_names = ["existing_bag_feat", "missing_bag_feat"]
    create_dummy_bag_files(
        "existing_bag_feat", features_path=features_dir, labels_path=labels_dir
    )
    # DO NOT create "missing_bag_feat" feature file

    dataset = ProcessedMILDataset(
        features_path=str(features_dir),
        labels_path=str(labels_dir),
        bag_names=bag_names,  # Explicitly include the missing bag
        bag_keys=["X", "Y"],  # Request mandatory keys
        load_at_init=False,  # Ensure it fails when __getitem__ is called
    )

    # Access existing bag should work
    _ = dataset[0]

    # Access missing bag should raise FileNotFoundError or ValueError
    with pytest.raises((FileNotFoundError, ValueError)):
        _ = dataset[1]


def test_fail_fast_missing_label_file(
    setup_full_dataset_paths, create_dummy_bag_files, tmp_path
):
    """
    Tests FileNotFoundError when a specific label file is missing.
    """
    features_dir = Path(setup_full_dataset_paths["features_path"])
    labels_dir = Path(setup_full_dataset_paths["labels_path"])

    bag_names = ["existing_bag_label", "missing_label_bag"]
    create_dummy_bag_files(
        "existing_bag_label", features_path=features_dir, labels_path=labels_dir
    )
    create_dummy_bag_files(
        "missing_label_bag", features_path=features_dir, labels_path=labels_dir
    )
    # Remove the label file for "missing_label_bag"
    (labels_dir / "missing_label_bag.npy").unlink()

    dataset = ProcessedMILDataset(
        features_path=str(features_dir),
        labels_path=str(labels_dir),
        bag_names=bag_names,
        bag_keys=["X", "Y"],  # Request mandatory keys
        load_at_init=False,
    )

    _ = dataset[0]
    with pytest.raises((FileNotFoundError, ValueError)):
        _ = dataset[1]


def test_fail_fast_missing_inst_label_file_if_path_provided(
    setup_full_dataset_paths, create_dummy_bag_files, tmp_path
):
    """
    Tests FileNotFoundError when instance label path is provided, but a file is missing.
    """
    features_dir = Path(setup_full_dataset_paths["features_path"])
    labels_dir = Path(setup_full_dataset_paths["labels_path"])
    inst_labels_dir = tmp_path / "inst_labels_specific"  # Use a new specific dir

    bag_names = ["bag_with_inst_label", "bag_missing_inst_label"]

    create_dummy_bag_files(
        "bag_with_inst_label",
        features_path=features_dir,
        labels_path=labels_dir,
        inst_labels_path=inst_labels_dir,
    )
    create_dummy_bag_files(
        "bag_missing_inst_label", features_path=features_dir, labels_path=labels_dir
    )  # Do not create inst label file here

    dataset = ProcessedMILDataset(
        features_path=str(features_dir),
        labels_path=str(labels_dir),
        inst_labels_path=str(
            inst_labels_dir
        ),  # Path is provided, so files are expected
        bag_names=bag_names,
        bag_keys=["X", "Y", "y_inst"],  # Request y_inst
        load_at_init=False,
    )

    name_to_idx = {name: i for i, name in enumerate(dataset.bag_names)}

    _ = dataset[name_to_idx["bag_with_inst_label"]]  # Should load successfully
    with pytest.raises((FileNotFoundError, ValueError)):
        _ = dataset[
            name_to_idx["bag_missing_inst_label"]
        ]  # Should fail because inst_labels_path was set and y_inst requested, but file is missing


def test_fail_fast_missing_coords_file_if_path_provided(
    setup_full_dataset_paths, create_dummy_bag_files, tmp_path
):
    """
    Tests FileNotFoundError when coords path is provided, but a file is missing.
    """
    features_dir = Path(setup_full_dataset_paths["features_path"])
    labels_dir = Path(setup_full_dataset_paths["labels_path"])
    coords_dir = tmp_path / "coords_specific"  # Use a new specific dir

    bag_names = ["bag_with_coords", "bag_missing_coords"]

    create_dummy_bag_files(
        "bag_with_coords",
        features_path=features_dir,
        labels_path=labels_dir,
        coords_path=coords_dir,
    )
    create_dummy_bag_files(
        "bag_missing_coords", features_path=features_dir, labels_path=labels_dir
    )  # Do not create coords file here

    dataset = ProcessedMILDataset(
        features_path=str(features_dir),
        labels_path=str(labels_dir),
        coords_path=str(coords_dir),  # Path is provided, so files are expected
        bag_names=bag_names,
        bag_keys=["X", "Y", "coords"],  # Request coords
        load_at_init=False,
    )

    name_to_idx = {name: i for i, name in enumerate(dataset.bag_names)}

    _ = dataset[name_to_idx["bag_with_coords"]]  # Should load successfully
    with pytest.raises((FileNotFoundError, ValueError)):
        _ = dataset[
            name_to_idx["bag_missing_coords"]
        ]  # Should fail because coords_path was set and coords requested, but file is missing


# # --- Other Functionality Tests ---


def test_len_method(setup_full_dataset_paths):
    """
    Tests the __len__ method.
    """
    dataset = ProcessedMILDataset(
        features_path=setup_full_dataset_paths["features_path"],
        labels_path=setup_full_dataset_paths["labels_path"],
        bag_keys=["X", "Y"],
        load_at_init=True,
    )
    assert len(dataset) == len(setup_full_dataset_paths["bag_names"])


def test_get_bag_labels_method(setup_full_dataset_paths):
    """
    Tests the get_bag_labels method.
    """
    dataset = ProcessedMILDataset(
        features_path=setup_full_dataset_paths["features_path"],
        labels_path=setup_full_dataset_paths["labels_path"],
        bag_keys=["X", "Y"],
        load_at_init=False,  # Don't load at init to test get_bag_labels explicitly loading
    )
    labels = dataset.get_bag_labels()
    assert isinstance(labels, list)
    assert len(labels) == len(setup_full_dataset_paths["bag_names"])
    assert isinstance(labels[0], np.ndarray)  # Should return numpy arrays


def test_get_bag_names_method(setup_full_dataset_paths):
    """
    Tests the get_bag_names method.
    """
    dataset = ProcessedMILDataset(
        features_path=setup_full_dataset_paths["features_path"],
        labels_path=setup_full_dataset_paths["labels_path"],
        bag_keys=["X", "Y"],
    )
    assert dataset.get_bag_names() == setup_full_dataset_paths["bag_names"]


def test_subset_method(setup_full_dataset_paths):
    """
    Tests the subset method.
    """
    dataset = ProcessedMILDataset(
        features_path=setup_full_dataset_paths["features_path"],
        labels_path=setup_full_dataset_paths["labels_path"],
        inst_labels_path=setup_full_dataset_paths["inst_labels_path"],
        coords_path=setup_full_dataset_paths["coords_path"],
        load_at_init=True,
    )

    original_bag_names = dataset.get_bag_names()
    subset_indices = [0, 2]  # Select bag_0 and bag_2

    subset_dataset = dataset.subset(subset_indices)

    assert len(subset_dataset) == len(subset_indices)
    assert subset_dataset.get_bag_names() == [
        original_bag_names[i] for i in subset_indices
    ]

    # Check if loaded bags are correctly transferred
    assert all(
        name in subset_dataset.loaded_bags for name in subset_dataset.get_bag_names()
    )
    assert all(
        name not in subset_dataset.loaded_bags
        for name in original_bag_names
        if name not in subset_dataset.get_bag_names()
    )


def test_adjacency_matrix_creation(setup_full_dataset_paths):
    """
    Tests that adjacency matrix is created when coords are present and requested.
    """
    dataset = ProcessedMILDataset(
        features_path=setup_full_dataset_paths["features_path"],
        labels_path=setup_full_dataset_paths["labels_path"],
        coords_path=setup_full_dataset_paths["coords_path"],  # Provide coords path
        bag_keys=["X", "Y", "adj"],  # Request adj
        load_at_init=True,
    )
    bag = dataset[0]
    assert "adj" in bag
    assert isinstance(bag["adj"], torch.Tensor)
    assert bag["adj"].is_sparse  # Check if it's a sparse tensor
    assert bag["adj"].shape[0] == bag["X"].shape[0]  # Adj size matches bag size


def test_adjacency_matrix_not_created_if_coords_not_in_bag_keys(
    setup_full_dataset_paths,
):
    """
    Tests that adjacency matrix is NOT created if 'adj' is not in bag_keys,
    even if coords are available.
    """
    dataset = ProcessedMILDataset(
        features_path=setup_full_dataset_paths["features_path"],
        labels_path=setup_full_dataset_paths["labels_path"],
        coords_path=setup_full_dataset_paths["coords_path"],
        bag_keys=["X", "Y"],  # 'adj' is NOT in bag_keys
        load_at_init=True,
    )
    bag = dataset[0]
    assert "adj" not in bag  # Should not be present


def test_tensordict_output(setup_full_dataset_paths):
    """
    Tests that __getitem__ returns a TensorDict.
    """
    dataset = ProcessedMILDataset(
        features_path=setup_full_dataset_paths["features_path"],
        labels_path=setup_full_dataset_paths["labels_path"],
        bag_keys=["X", "Y"],
        load_at_init=True,
    )
    bag = dataset[0]
    assert isinstance(bag, TensorDict)
    assert isinstance(bag["X"], torch.Tensor)
    assert isinstance(bag["Y"], torch.Tensor)
