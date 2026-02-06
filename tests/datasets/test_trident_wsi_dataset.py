import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

from torchmil.datasets import TridentWSIDataset

# --- Fixtures ---

@pytest.fixture
def base_kwargs():
    """Standard arguments for initializing the dataset."""
    return {
        "base_path": "/mock/base",
        "labels_path": "/mock/labels",
        "feature_extractor": "conch_v15",
        "magnification": 20,
        "patch_size": 512,
        "overlap_pixels": 0,
        "wsi_names": ["slide_1", "slide_2"]
    }

# --- Tests ---

@patch("torchmil.datasets.wsi_dataset.WSIDataset.__init__", return_value=None)
def test_init_defaults(mock_super_init, base_kwargs):
    ds = TridentWSIDataset(**base_kwargs)

    assert ds.patch_size == 512
    assert ds.magnification == 20
    assert ds.overlap_pixels == 0
    assert ds.trident_folder == "20x_512px_0px_overlap/"
    
    expected_features_path = "/mock/base20x_512px_0px_overlap/features_conch_v15/"
    expected_coords_path = "/mock/base20x_512px_0px_overlap/patches/"
    
    mock_super_init.assert_called_once()
    call_kwargs = mock_super_init.call_args[1]
    
    assert call_kwargs["features_path"] == expected_features_path
    assert call_kwargs["coords_path"] == expected_coords_path
    assert call_kwargs["dist_thr"] == pytest.approx(np.sqrt(2.0))

@patch("torchmil.datasets.wsi_dataset.WSIDataset.__init__", return_value=None)
def test_init_custom_threshold(mock_super_init, base_kwargs):
    kwargs = base_kwargs.copy()
    kwargs["dist_thr"] = 5.5
    
    # Fix: Assign to '_' to silence the "unused variable" error
    # while still ensuring TridentWSIDataset initializes.
    _ = TridentWSIDataset(**kwargs)
    
    call_kwargs = mock_super_init.call_args[1]
    assert call_kwargs["dist_thr"] == 5.5

@patch("torchmil.datasets.wsi_dataset.WSIDataset._load_labels")
@patch("os.path.isdir")
@patch("torchmil.datasets.wsi_dataset.WSIDataset.__init__", return_value=None)
def test_load_labels_directory_mode(mock_super_init, mock_isdir, mock_super_load, base_kwargs):
    """Test directory mode logic. Manually set labels_path since super().__init__ is mocked."""
    mock_isdir.return_value = True
    
    ds = TridentWSIDataset(**base_kwargs)
    ds.labels_path = base_kwargs["labels_path"]  # Manually set attribute usually set by super()
    
    expected_label = np.array([0])
    mock_super_load.return_value = expected_label
    
    result = ds._load_labels("slide_1")
    
    assert result == expected_label
    mock_isdir.assert_called_with("/mock/labels")
    mock_super_load.assert_called_with("slide_1")

@patch("pandas.read_csv")
@patch("os.path.isdir")
@patch("torchmil.datasets.wsi_dataset.WSIDataset.__init__", return_value=None)
def test_load_labels_csv_mode_success(mock_super_init, mock_isdir, mock_read_csv, base_kwargs):
    """Test CSV mode logic. Manually set labels_path."""
    mock_isdir.return_value = False
    
    df = pd.DataFrame({
        "filename": ["slide_1", "slide_2"],
        "grade": [0, 1]
    })
    mock_read_csv.return_value = df
    
    kwargs = base_kwargs.copy()
    kwargs["wsi_name_col"] = "filename"
    kwargs["wsi_label_col"] = "grade"
    
    ds = TridentWSIDataset(**kwargs)
    ds.labels_path = kwargs["labels_path"] # Manually set attribute
    
    label = ds._load_labels("slide_2")
    
    assert label[0] == 1
    mock_read_csv.assert_called_once_with("/mock/labels")

@patch("pandas.read_csv")
@patch("os.path.isdir")
@patch("torchmil.datasets.wsi_dataset.WSIDataset.__init__", return_value=None)
def test_load_labels_csv_missing_kwargs(mock_super_init, mock_isdir, mock_read_csv, base_kwargs):
    mock_isdir.return_value = False
    mock_read_csv.return_value = pd.DataFrame()
    
    ds = TridentWSIDataset(**base_kwargs)
    ds.labels_path = base_kwargs["labels_path"] # Manually set attribute
    
    with pytest.raises(ValueError, match="must provide 'wsi_name_col' and 'wsi_label_col'"):
        ds._load_labels("slide_1")

@patch("pandas.read_csv")
@patch("os.path.isdir")
@patch("torchmil.datasets.wsi_dataset.WSIDataset.__init__", return_value=None)
def test_load_labels_csv_not_found(mock_super_init, mock_isdir, mock_read_csv, base_kwargs):
    mock_isdir.return_value = False
    
    # Mock DF to raise ValueError when accessing specific data
    mock_df = MagicMock()
    mock_df.loc.__getitem__.side_effect = ValueError("Forced error")
    mock_read_csv.return_value = mock_df

    kwargs = base_kwargs.copy()
    kwargs["wsi_name_col"] = "name"
    kwargs["wsi_label_col"] = "label"
    
    ds = TridentWSIDataset(**kwargs)
    ds.labels_path = kwargs["labels_path"] # Manually set attribute
    
    with pytest.raises(ValueError, match="Could not read the label"):
        ds._load_labels("slide_X")

@patch("h5py.File")
@patch("torchmil.datasets.wsi_dataset.WSIDataset.__init__", return_value=None)
def test_load_coords_calculation(mock_super_init, mock_h5, base_kwargs):
    ds = TridentWSIDataset(**base_kwargs)
    ds.coords_path = "/mock/patches/"
    ds.file_type = ".h5"
    ds.patch_size = 512
    
    raw_coords = np.array([
        [1024, 2048],
        [1536, 2560]
    ])
    
    mock_file = MagicMock()
    mock_file.__getitem__.return_value = raw_coords
    mock_h5.return_value = mock_file
    
    expected_coords = np.array([[0, 0], [1, 1]])
    
    result = ds._load_coords("slide_1")
    
    mock_h5.assert_called_with("/mock/patches/slide_1_patches.h5", "r")
    np.testing.assert_array_equal(result, expected_coords)

@patch("h5py.File")
@patch("torchmil.datasets.wsi_dataset.WSIDataset.__init__", return_value=None)
def test_load_coords_none(mock_super_init, mock_h5, base_kwargs):
    """
    Test _load_coords when the H5 file slice returns None.
    Structure: h5py.File(...)['coords'][:] -> None
    """
    ds = TridentWSIDataset(**base_kwargs)
    ds.coords_path = "/mock/patches/"
    ds.file_type = ".h5"
    
    # 1. Mock the File object
    mock_file_obj = MagicMock()
    
    # 2. Mock the dataset object returned by file['coords']
    mock_dataset = MagicMock()
    
    # 3. Mock the slice operator [:] on the dataset to return None
    mock_dataset.__getitem__.return_value = None
    
    # 4. Connect them
    mock_file_obj.__getitem__.return_value = mock_dataset
    mock_h5.return_value = mock_file_obj
    
    result = ds._load_coords("slide_empty")
    assert result is None