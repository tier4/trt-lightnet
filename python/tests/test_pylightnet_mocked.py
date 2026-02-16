"""Test pylightnet package with mocked C++ libraries for non-GPU environments."""

import ctypes
import os
import tempfile
import unittest.mock as mock

import pytest


class TestPylightnetMocked:
    """Test pylightnet functionality with mocked C++ dependencies."""

    @pytest.fixture(autouse=True)
    def setup_mocks(self, monkeypatch):
        """Set up mocks for C++ library dependencies."""
        # Create mock library object
        mock_lib = mock.MagicMock()

        # Mock essential C functions
        mock_lib.create_trt_lightnet.return_value = 0x12345  # Dummy pointer
        mock_lib.destroy_trt_lightnet.return_value = None

        # Mock get_input_size to set output parameters
        def mock_get_input_size(instance, batch, chan, height, width):
            # Dereference the pointers and set values
            batch._obj.value = 1
            chan._obj.value = 3
            height._obj.value = 640
            width._obj.value = 640

        mock_lib.trt_lightnet_get_input_size.side_effect = mock_get_input_size

        # Mock bbox array functions
        def mock_get_bbox_array(inst, size):
            size._obj.value = 0  # Set size to 0 (no bboxes)
            return None

        mock_lib.get_bbox_array.side_effect = mock_get_bbox_array

        # Patch ctypes.CDLL to return our mock for library files
        original_cdll = ctypes.CDLL

        def mock_cdll(path, **kwargs):
            if isinstance(path, str) and (
                "libcnpy.so" in path or "liblightnetinfer.so" in path
            ):
                return mock_lib
            return original_cdll(path, **kwargs)

        monkeypatch.setattr(ctypes, "CDLL", mock_cdll)

        # Store for use in tests
        self.mock_lib = mock_lib

    def test_load_config(self):
        """Test configuration file loading."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("# Test config\n")
            f.write("--onnx=model.onnx\n")
            f.write("--c=80\n")
            f.write("--thresh=0.5\n")
            f.write("--precision=fp16\n")
            f.write("--anchors=10,20,30,40\n")
            f.write("--sparse=true\n")
            config_path = f.name

        try:
            # Import here to ensure mocks are active
            import pylightnet

            config = pylightnet.load_config(config_path)

            assert config["onnx"] == "model.onnx"
            assert config["c"] == 80
            assert config["thresh"] == 0.5
            assert config["precision"] == "fp16"
            assert config["anchors"] == [10, 20, 30, 40]
            assert config["sparse"] is True
        finally:
            os.unlink(config_path)

    def test_load_names_from_file(self):
        """Test loading class names from file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("person\n")
            f.write("car\n")
            f.write("bus\n")
            f.write("\n")  # Empty line should be ignored
            f.write("truck\n")
            names_path = f.name

        try:
            import pylightnet

            names = pylightnet.load_names_from_file(names_path)

            assert names == ["person", "car", "bus", "truck"]
        finally:
            os.unlink(names_path)

    def test_load_colormap_from_file(self):
        """Test loading colormap from file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("255,0,0\n")  # Red
            f.write("0,255,0\n")  # Green
            f.write("0,0,255\n")  # Blue
            f.write("\n")  # Empty line
            f.write("255,255,0\n")  # Yellow
            colormap_path = f.name

        try:
            import pylightnet

            colormap = pylightnet.load_colormap_from_file(colormap_path)

            expected = [255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 0]
            assert colormap == expected
        finally:
            os.unlink(colormap_path)

    def test_compute_iou(self):
        """Test IoU computation."""
        import pylightnet

        # Test case 1: Perfect overlap
        box1 = [0, 0, 10, 10]
        box2 = [0, 0, 10, 10]
        iou = pylightnet.compute_iou(box1, box2)
        assert iou == 1.0

        # Test case 2: No overlap
        box1 = [0, 0, 10, 10]
        box2 = [20, 20, 30, 30]
        iou = pylightnet.compute_iou(box1, box2)
        assert iou == 0.0

        # Test case 3: Partial overlap
        box1 = [0, 0, 10, 10]
        box2 = [5, 5, 15, 15]
        iou = pylightnet.compute_iou(box1, box2)
        # Intersection: 5x5 = 25
        # Union: 100 + 100 - 25 = 175
        # IoU: 25/175 = 0.142857...
        assert abs(iou - 25 / 175) < 1e-6

    def test_nms_bboxes(self):
        """Test Non-Maximum Suppression."""
        import pylightnet

        # Create test bounding boxes with overlaps
        bboxes = [
            {"box": [0, 0, 10, 10], "prob": 0.9},
            {"box": [2, 2, 12, 12], "prob": 0.8},  # High overlap with first
            {"box": [20, 20, 30, 30], "prob": 0.7},  # No overlap
            {"box": [22, 22, 32, 32], "prob": 0.6},  # High overlap with third
        ]

        # Apply NMS with IoU threshold 0.5
        result = pylightnet.nms_bboxes(bboxes, iou_thresh=0.5)

        # The actual behavior depends on the IoU values
        # For boxes [0,0,10,10] and [2,2,12,12]: IoU = 64/156 = 0.41 < 0.5, so both kept
        # For boxes [20,20,30,30] and [22,22,32,32]: IoU = 64/156 = 0.41 < 0.5, so both kept
        assert len(result) == 4  # All boxes are kept since IoU < 0.5

    def test_merge_bbox(self):
        """Test merging bounding boxes from different sources."""
        import pylightnet

        # First set of boxes
        bboxes1 = [
            {"box": [0, 0, 10, 10], "prob": 0.9, "label": 0},
            {"box": [20, 20, 30, 30], "prob": 0.7, "label": 1},
        ]

        # Second set of boxes (different label space)
        bboxes2 = [
            {"box": [50, 50, 60, 60], "prob": 0.8, "label": 0},  # 'car' in names2
            {"box": [70, 70, 80, 80], "prob": 0.6, "label": 1},  # 'truck' in names2
        ]

        names1 = ["person", "car", "bus"]
        names2 = ["car", "truck"]

        # Merge bboxes2 into bboxes1
        result = pylightnet.merge_bbox(bboxes1, bboxes2, names1, names2)

        # Should have 3 boxes total (2 original + 1 matched 'car')
        assert len(result) == 3
        # Check that 'car' from bboxes2 was remapped to label 1 in names1
        car_boxes = [b for b in result if b["label"] == 1]
        assert len(car_boxes) == 2  # Original car + merged car

    def test_model_config_parsing(self):
        """Test parsing model configuration."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("person\ncar\n")
            names_path = f.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("255,0,0\n0,255,0\n")
            rgb_path = f.name

        try:
            import pylightnet

            config_dict = {
                "onnx": "model.onnx",
                "c": 2,
                "thresh": 0.5,
                "anchors": [10, 20, 30, 40],
                "num_anchors": 2,
                "names": names_path,
                "rgb": rgb_path,
            }

            model_config = pylightnet.parse_model_config(config_dict)

            assert model_config.model_path == b"model.onnx"
            assert model_config.num_class == 2
            assert model_config.score_threshold == 0.5
            assert model_config.num_anchors == 2
            assert abs(model_config.nms_threshold - 0.45) < 1e-6  # Default value
        finally:
            os.unlink(names_path)
            os.unlink(rgb_path)

    def test_inference_config_parsing(self):
        """Test parsing inference configuration."""
        import pylightnet

        config_dict = {
            "precision": "fp16",
            "sparse": True,
            "calibration_images": "/path/to/images",
            "calib": "Entropy",
        }

        inference_config = pylightnet.parse_inference_config(config_dict)

        assert inference_config.precision == b"fp16"
        assert inference_config.sparse is True
        assert inference_config.calibration_images == b"/path/to/images"
        assert inference_config.calibration_type == b"Entropy"
        assert inference_config.batch_size == 1
        assert inference_config.workspace_size == 1073741824  # 1GB default

    def test_build_config_parsing(self):
        """Test parsing build configuration."""
        import pylightnet

        config_dict = {
            "sparse": True,
        }

        build_config = pylightnet.parse_build_config(config_dict)

        assert build_config.sparse is True
        assert build_config.calib_type_str == b"Entropy"
        assert build_config.dla_core_id == -1
        assert build_config.profile_per_layer is True

    def test_trt_lightnet_initialization(self):
        """Test TrtLightnet class initialization with mocks."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("person\n")
            names_path = f.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("255,0,0\n")
            rgb_path = f.name

        try:
            import pylightnet

            config_dict = {
                "onnx": "model.onnx",
                "c": 1,
                "thresh": 0.5,
                "names": names_path,
                "rgb": rgb_path,
                "precision": "fp16",
                "calibration_images": "/path",
                "calib": "Entropy",
            }

            # This should work with our mocked libraries
            lightnet = pylightnet.create_lightnet_from_config(config_dict)

            # Test get_input_size with mocked values
            batch, chan, height, width = lightnet.get_input_size()
            assert batch == 1
            assert chan == 3
            assert height == 640
            assert width == 640

            # Test get_bboxes (should return empty list with our mock)
            bboxes = lightnet.get_bboxes()
            assert bboxes == []

            # Clean up
            lightnet.destroy()

        finally:
            os.unlink(names_path)
            os.unlink(rgb_path)
