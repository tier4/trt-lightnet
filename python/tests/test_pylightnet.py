"""Test pylightnet package with GPU/C++ libraries."""

# Avoid loading local pylightnet
from __future__ import absolute_import

import ctypes
import os

import cv2
import numpy as np
import pylightnet
import pytest


def test_load_libs():
    """Test loading libraries."""
    # Load dependent library libcnpy.so as a global library
    libcnpy_path = os.path.join(os.path.dirname(pylightnet.__file__), "libcnpy.so")
    if not os.path.exists(libcnpy_path):
        pytest.skip(f"Library libcnpy.so not found at {libcnpy_path}")

    try:
        ctypes.CDLL(libcnpy_path, mode=ctypes.RTLD_GLOBAL)
    except Exception as e:
        pytest.skip(f"Failed to load dependent library libcnpy.so: {e}")

    # Load the main library liblightnetinfer.so
    lib_path = os.path.join(os.path.dirname(pylightnet.__file__), "liblightnetinfer.so")
    if not os.path.exists(lib_path):
        pytest.skip(f"Library liblightnetinfer.so not found at {lib_path}")

    try:
        ctypes.CDLL(lib_path)
    except Exception as e:
        pytest.skip(f"Failed to load library liblightnetinfer.so: {e}")


@pytest.mark.parametrize(
    "config_path",
    [
        "assets/trt-lightnet/20250520_co-mlops-large-anonymization-pipeline-v0.0.3/co-mlops-large-anonymization-pipeline-v0.0.3.txt"
    ],
)
@pytest.mark.parametrize(
    "image_path",
    ["assets/images/sample_01.jpg"],
)
def test_inference(config_path: str, image_path: str):
    """Test inference."""
    # Check if libraries are available
    libcnpy_path = os.path.join(os.path.dirname(pylightnet.__file__), "libcnpy.so")
    lib_path = os.path.join(os.path.dirname(pylightnet.__file__), "liblightnetinfer.so")

    if not os.path.exists(libcnpy_path) or not os.path.exists(lib_path):
        pytest.skip("Required C++ libraries not found")

    if not os.path.exists(config_path):
        pytest.skip(f"Config file {config_path} not found.")
    if not os.path.exists(image_path):
        pytest.skip(f"Image file {image_path} not found.")

    # Load the model
    config_dict = pylightnet.load_config(config_path)
    onnx_path = config_dict["onnx"]
    if not os.path.exists(onnx_path):
        pytest.skip(f"ONNX file {onnx_path} not found.")

    try:
        lightnet = pylightnet.create_lightnet_from_config(config_dict)
    except Exception as e:
        if "Failed to load" in str(e):
            pytest.skip(f"Failed to load libraries: {e}")
        raise
    names = pylightnet.load_names_from_file(config_dict["names"])
    colormap = pylightnet.load_colormap_from_file(config_dict["rgb"])

    # Load the image and run inference
    image = cv2.imread(image_path)
    orig_img = image.copy()
    lightnet.infer(image, cuda=True)
    bboxes = lightnet.get_bboxes()
    pylightnet.draw_bboxes_on_image(image, bboxes, colormap, names)

    assert not np.array_equal(orig_img, image)
    # Clean up
    lightnet.destroy()


def skip_if_no_gpu_libs():
    """Skip test if GPU libraries are not available."""
    libcnpy_path = os.path.join(os.path.dirname(pylightnet.__file__), "libcnpy.so")
    lib_path = os.path.join(os.path.dirname(pylightnet.__file__), "liblightnetinfer.so")

    if not os.path.exists(libcnpy_path) or not os.path.exists(lib_path):
        pytest.skip("Required C++ libraries not found")


class TestTrtLightnetGPU:
    """Test TrtLightnet functionality with actual GPU/C++ libraries."""

    def setup_method(self):
        """Check for GPU libraries before each test."""
        skip_if_no_gpu_libs()

    def load_test_config(
        self,
        config_path="assets/trt-lightnet/20250520_co-mlops-large-anonymization-pipeline-v0.0.3/co-mlops-large-anonymization-pipeline-v0.0.3.txt",
    ):
        """Load actual configuration from asset files."""

        if not os.path.exists(config_path):
            pytest.skip(f"Config file {config_path} not found.")

        config_dict = pylightnet.load_config(config_path)
        return config_dict

    def test_trt_lightnet_initialization_and_cleanup(self):
        """Test proper initialization and cleanup of TrtLightnet."""
        config_dict = self.load_test_config()

        try:
            # Skip if ONNX file doesn't exist
            if not os.path.exists(config_dict["onnx"]):
                pytest.skip(f"ONNX file {config_dict['onnx']} not found")

            lightnet = pylightnet.create_lightnet_from_config(config_dict)

            # Test that instance was created
            assert lightnet.instance is not None
            assert lightnet.instance != 0

            # Test input size retrieval
            batch, chan, height, width = lightnet.get_input_size()
            assert batch > 0
            assert chan in [1, 3]  # Grayscale or RGB
            assert height > 0
            assert width > 0

            # Clean up properly
            lightnet.destroy()

        except RuntimeError as e:
            if "Failed to load" in str(e) or "Failed to create" in str(e):
                pytest.skip(f"GPU initialization failed: {e}")
            raise

    def test_batch_inference(self):
        """Test batch inference functionality."""
        # Use the anonymization pipeline config which has subnet configuration
        config_dict = self.load_test_config(
            "assets/trt-lightnet/20250520_co-mlops-large-anonymization-pipeline-v0.0.3/co-mlops-large-anonymization-pipeline-v0.0.3.txt"
        )

        try:
            if not os.path.exists(config_dict["onnx"]) or not os.path.exists(
                config_dict.get("subnet_onnx", "")
            ):
                pytest.skip("Required ONNX files not found")

            lightnet = pylightnet.create_lightnet_from_config(config_dict)

            # Create test image
            image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

            # Create fake bounding boxes for batch processing
            test_bboxes = [
                {"box": [10, 10, 100, 100], "label": 0, "prob": 0.9},
                {"box": [200, 200, 300, 300], "label": 1, "prob": 0.8},
                {"box": [400, 400, 500, 500], "label": 0, "prob": 0.7},
            ]

            names = pylightnet.load_names_from_file(config_dict["names"])

            # Test batch inference
            results = lightnet.infer_subnet_batches_from_bboxes(
                test_bboxes,
                image,
                names,
                ["PEDESTRIAN", "CAR"],  # target labels from config
                ["LICENSE_PLATE", "HUMAN_HEAD"],  # subnet labels from config
                batch_size=2,
                min_crop_size=50,
                debug=False,
            )

            assert isinstance(results, list)

            lightnet.destroy()

        except RuntimeError as e:
            if "Failed to load" in str(e) or "Failed to create" in str(e):
                pytest.skip(f"GPU initialization failed: {e}")
            raise

    def test_segmentation_masks(self):
        """Test segmentation mask generation."""
        config_dict = self.load_test_config()

        try:
            if not os.path.exists(config_dict["onnx"]):
                pytest.skip(f"ONNX file {config_dict['onnx']} not found")

            lightnet = pylightnet.create_lightnet_from_config(config_dict)

            # Create segmentation data
            seg_data = {
                0: {"name": "background", "r": 0, "g": 0, "b": 0, "dynamic": False},
                1: {"name": "road", "r": 128, "g": 64, "b": 128, "dynamic": False},
                2: {"name": "car", "r": 0, "g": 0, "b": 142, "dynamic": True},
            }

            # Convert to argmax2bgr
            argmax2bgr_ptr = lightnet.segmentation_to_argmax2bgr(seg_data)

            # Make mask
            lightnet.make_mask(argmax2bgr_ptr)

            # Get masks
            masks = lightnet.get_masks_from_cpp()

            assert isinstance(masks, list)

            # Clean up
            lightnet.free_argmax2bgr()
            lightnet.destroy()

        except RuntimeError as e:
            if "Failed to load" in str(e) or "Failed to create" in str(e):
                pytest.skip(f"GPU initialization failed: {e}")
            raise

    def test_multi_stage_inference(self):
        """Test multi-stage inference functionality."""
        # Use the anonymization pipeline config which has subnet configuration
        config_dict = self.load_test_config(
            "assets/trt-lightnet/20250520_co-mlops-large-anonymization-pipeline-v0.0.3/co-mlops-large-anonymization-pipeline-v0.0.3.txt"
        )

        try:
            if not os.path.exists(config_dict["onnx"]) or not os.path.exists(
                config_dict.get("subnet_onnx", "")
            ):
                pytest.skip("Required ONNX files not found")

            lightnet = pylightnet.create_lightnet_from_config(config_dict)

            # Create test image
            image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

            # Run multi-stage inference
            target_list = ["PEDESTRIAN", "CAR"]
            lightnet.infer_multi_stage(image, target_list, cuda=True)

            # Get results from both stages
            main_bboxes = lightnet.get_bboxes()
            subnet_bboxes = lightnet.get_subnet_bboxes()

            assert isinstance(main_bboxes, list)
            assert isinstance(subnet_bboxes, list)

            lightnet.destroy()

        except RuntimeError as e:
            if "Failed to load" in str(e) or "Failed to create" in str(e):
                pytest.skip(f"GPU initialization failed: {e}")
            raise

    def test_performance_metrics(self):
        """Test performance measurement with profiling."""
        config_dict = self.load_test_config()

        # Enable profiling
        config_dict["profile"] = True

        try:
            if not os.path.exists(config_dict["onnx"]):
                pytest.skip(f"ONNX file {config_dict['onnx']} not found")

            inference_config = pylightnet.parse_inference_config(config_dict)
            inference_config.profile = True

            model_config = pylightnet.parse_model_config(config_dict)
            build_config = pylightnet.parse_build_config(config_dict)

            lightnet = pylightnet.TrtLightnet(
                model_config, inference_config, build_config
            )

            # Create test image
            image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

            # Run inference multiple times for profiling
            import time

            times = []

            for _ in range(10):
                start = time.time()
                lightnet.infer(image, cuda=True)
                lightnet.get_bboxes()
                end = time.time()
                times.append(end - start)

            # Verify timing consistency (after warmup)
            avg_time = sum(times[2:]) / len(times[2:])  # Skip first 2 for warmup
            assert avg_time > 0

            lightnet.destroy()

        except RuntimeError as e:
            if "Failed to load" in str(e) or "Failed to create" in str(e):
                pytest.skip(f"GPU initialization failed: {e}")
            raise
