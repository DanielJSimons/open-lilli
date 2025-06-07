import unittest
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path
import sys

# Ensure the source directory is in the path for imports
# This might need adjustment based on actual project structure
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from open_lilli.corporate_asset_library import CorporateAssetLibrary
from open_lilli.models import AssetLibraryConfig

class TestCorporateAssetLibrary(unittest.TestCase):

    def setUp(self):
        self.mock_config = AssetLibraryConfig(
            dam_api_url="https://fake-dam.com/api",
            api_key="test_api_key",
            max_asset_size_mb=10,
            preferred_asset_types=["icon", "photo", "logo"],
            brand_guidelines_strict=False,
            fallback_to_external=True
        )
        self.cal = CorporateAssetLibrary(config=self.mock_config)
        # Ensure cache_dir is mocked or handled appropriately if tests write files
        self.cal.cache_dir = Path("test_cache")
        self.cal.cache_dir.mkdir(exist_ok=True)

    def tearDown(self):
        # Clean up test_cache directory if needed
        if self.cal.cache_dir.exists():
            # Ensure all files are unlinked before attempting to remove directory
            for item in self.cal.cache_dir.iterdir():
                if item.is_file(): # Only unlink files
                    item.unlink()
                elif item.is_dir(): # If there are subdirs, handle them (though not expected for this cache)
                    pass
            try:
                self.cal.cache_dir.rmdir()
            except OSError as e:
                print(f"Warning: Could not remove test_cache directory: {e}")


    @patch('requests.Session.get')
    def test_search_assets_metadata_filters(self, mock_get):
        """Test that search_assets passes metadata filters to the API call."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'assets': []}
        mock_get.return_value = mock_response

        test_query = "test query"
        test_orientation = "landscape"
        test_color = "blue"
        test_tags = ["tag1", "tag2"]

        self.cal.search_assets(
            query=test_query,
            orientation=test_orientation,
            dominant_color=test_color,
            tags=test_tags
        )

        mock_get.assert_called_once()
        called_args, called_kwargs = mock_get.call_args
        self.assertEqual(called_args[0], "https://fake-dam.com/api/search")

        expected_params = {
            'q': test_query,
            'type': 'image', # default
            'limit': 10,     # default
            'format': 'json',
            'orientation': test_orientation,
            'dominant_color': test_color,
            'tags': 'tag1,tag2'
        }
        self.assertEqual(called_kwargs['params'], expected_params)

    @patch('requests.Session.get')
    def test_search_assets_no_optional_filters(self, mock_get):
        """Test search_assets without optional filters."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'assets': []}
        mock_get.return_value = mock_response

        test_query = "simple query"
        self.cal.search_assets(query=test_query, asset_type="icon", limit=5)

        mock_get.assert_called_once()
        called_args, called_kwargs = mock_get.call_args
        self.assertEqual(called_args[0], "https://fake-dam.com/api/search")

        expected_params = {
            'q': test_query,
            'type': 'icon',
            'limit': 5,
            'format': 'json',
            'preferred': 'true' # Since 'icon' is in preferred_asset_types
        }
        self.assertEqual(called_kwargs['params'], expected_params)

    @patch('open_lilli.corporate_asset_library.Image') # Patch PIL.Image
    def test_process_image_with_pillow_cropping(self, MockImage):
        """Test the cropping logic in _process_image_with_pillow."""
        mock_img_instance = MagicMock()
        MockImage.open.return_value = mock_img_instance
        mock_img_instance.format = "PNG" # So it saves as PNG

        # Test case 1: Image is wider than target
        mock_img_instance.size = (2000, 1000) # current_aspect_ratio = 2.0
        target_aspect_ratio_wider = 1.6 # e.g. 16:10

        # Expected new width = 1.6 * 1000 = 1600
        # left = (2000 - 1600) // 2 = 200
        # top = 0
        # right = 200 + 1600 = 1800
        # bottom = 1000
        expected_crop_box_wider = (200, 0, 1800, 1000)

        # Create a dummy image file for the test
        dummy_image_path = self.cal.cache_dir / "wider_image.png"
        with open(dummy_image_path, "w") as f: # create empty file, content doesn't matter due to mock
            f.write("dummy")

        processed_path_wider = self.cal._process_image_with_pillow(dummy_image_path, target_aspect_ratio_wider)
        self.assertIsNotNone(processed_path_wider)
        mock_img_instance.crop.assert_called_with(expected_crop_box_wider)
        mock_img_instance.save.assert_called()
        self.assertEqual(processed_path_wider.name, "wider_image_processed.png")

        # Test case 2: Image is taller than target
        mock_img_instance.reset_mock() # Reset mocks for next call
        MockImage.open.return_value = mock_img_instance # Re-assign after reset
        mock_img_instance.size = (1000, 2000) # current_aspect_ratio = 0.5
        target_aspect_ratio_taller = 1.0 # e.g. 1:1

        # Expected new height = 1000 / 1.0 = 1000
        # left = 0
        # top = (2000 - 1000) // 2 = 500
        # right = 1000
        # bottom = 500 + 1000 = 1500
        expected_crop_box_taller = (0, 500, 1000, 1500)

        dummy_image_path_taller = self.cal.cache_dir / "taller_image.png"
        with open(dummy_image_path_taller, "w") as f:
            f.write("dummy")

        processed_path_taller = self.cal._process_image_with_pillow(dummy_image_path_taller, target_aspect_ratio_taller)
        self.assertIsNotNone(processed_path_taller)
        mock_img_instance.crop.assert_called_with(expected_crop_box_taller)
        mock_img_instance.save.assert_called()

        # Test case 3: Image aspect ratio matches target (within tolerance)
        mock_img_instance.reset_mock()
        MockImage.open.return_value = mock_img_instance
        mock_img_instance.size = (1600, 1000) # current_aspect_ratio = 1.6
        target_aspect_ratio_match = 1.605 # within 0.01 tolerance

        dummy_image_path_match = self.cal.cache_dir / "match_image.png"
        with open(dummy_image_path_match, "w") as f:
            f.write("dummy")

        processed_path_match = self.cal._process_image_with_pillow(dummy_image_path_match, target_aspect_ratio_match)
        self.assertIsNotNone(processed_path_match)
        mock_img_instance.crop.assert_not_called() # Should not crop if within tolerance and logic skips
                                                 # Current logic crops anyway, so this might need adjustment
                                                 # For now, current logic always re-saves, so crop might be called with full dimensions
                                                 # Let's adjust the test for current behavior: crop is called with (0,0,width,height)
        # The logic in _process_image_with_pillow was: `cropped_img = img` if tolerance is met.
        # So, `crop` itself is not called on `img`. The original `img` (mock_img_instance) is then saved.
        # If the logic was `cropped_img = img.crop((0,0,width,height))`, then crop would be called.
        # Given the current code: `cropped_img = img`
        # So, the assertion `mock_img_instance.crop.assert_not_called()` is correct.
        mock_img_instance.save.assert_called_once() # Saved, but not cropped.

    @patch.object(CorporateAssetLibrary, 'search_assets')
    @patch.object(CorporateAssetLibrary, 'download_asset')
    @patch.object(CorporateAssetLibrary, '_process_image_with_pillow')
    def test_get_brand_approved_image_calls_processing(
        self, mock_process_image, mock_download_asset, mock_search_assets
    ):
        """Test get_brand_approved_image calls _process_image_with_pillow if target_aspect_ratio is provided."""
        mock_search_assets.return_value = [{'id': 'asset1', 'name': 'Test Asset'}]

        dummy_downloaded_path = self.cal.cache_dir / "downloaded.png"
        with open(dummy_downloaded_path, "w") as f: f.write("dummy")
        mock_download_asset.return_value = dummy_downloaded_path

        processed_image_path = self.cal.cache_dir / "processed.png"
        mock_process_image.return_value = processed_image_path

        target_ratio = 16/9
        result_path = self.cal.get_brand_approved_image(
            query="test",
            slide_index=0,
            target_aspect_ratio=target_ratio
        )

        mock_search_assets.assert_called_once_with(
            "test", asset_type="image", limit=5,
            orientation=None, dominant_color=None, tags=None
        )
        mock_download_asset.assert_called_once_with({'id': 'asset1', 'name': 'Test Asset'})
        mock_process_image.assert_called_once_with(dummy_downloaded_path, target_ratio)
        self.assertEqual(result_path, processed_image_path)

    @patch.object(CorporateAssetLibrary, 'search_assets')
    @patch.object(CorporateAssetLibrary, 'download_asset')
    @patch.object(CorporateAssetLibrary, '_process_image_with_pillow')
    def test_get_brand_approved_image_no_processing_if_no_ratio(
        self, mock_process_image, mock_download_asset, mock_search_assets
    ):
        """Test get_brand_approved_image does NOT call _process_image_with_pillow if no ratio is given."""
        mock_search_assets.return_value = [{'id': 'asset1'}]
        dummy_downloaded_path = self.cal.cache_dir / "downloaded_no_ratio.png"
        with open(dummy_downloaded_path, "w") as f: f.write("dummy")
        mock_download_asset.return_value = dummy_downloaded_path

        result_path = self.cal.get_brand_approved_image(query="test", slide_index=0, target_aspect_ratio=None)

        mock_download_asset.assert_called_once()
        mock_process_image.assert_not_called()
        self.assertEqual(result_path, dummy_downloaded_path)

    @patch.object(CorporateAssetLibrary, 'search_assets')
    @patch.object(CorporateAssetLibrary, 'download_asset')
    @patch.object(CorporateAssetLibrary, '_process_image_with_pillow')
    def test_get_brand_approved_image_fallback_on_processing_failure(
        self, mock_process_image, mock_download_asset, mock_search_assets
    ):
        """Test get_brand_approved_image returns original image if processing fails."""
        mock_search_assets.return_value = [{'id': 'asset1'}]
        dummy_downloaded_path = self.cal.cache_dir / "downloaded_processing_fail.png"
        with open(dummy_downloaded_path, "w") as f: f.write("dummy")
        mock_download_asset.return_value = dummy_downloaded_path

        mock_process_image.return_value = None # Simulate processing failure

        with self.assertLogs(logger='open_lilli.corporate_asset_library', level='WARNING') as log_watcher:
            result_path = self.cal.get_brand_approved_image(query="test", slide_index=0, target_aspect_ratio=1.0)

        self.assertEqual(result_path, dummy_downloaded_path)
        mock_process_image.assert_called_once()
        # Check for warning log
        self.assertTrue(any("Image processing failed" in message for message in log_watcher.output))


if __name__ == '__main__':
    unittest.main()
