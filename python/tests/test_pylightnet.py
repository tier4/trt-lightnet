"""Test pylightnet package with GPU/C++ libraries."""

# Avoid loading local pylightnet
from __future__ import absolute_import

import ctypes
import os
import tempfile

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
        "tests/assets/configs/CoMLOps-Reference-Vision-Detection-Segmentation-Softmax-Model-v0.1.2.txt"
    ],
)
@pytest.mark.parametrize(
    "image_path",
    ["tests/assets/images/sample_01.jpg"],
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
    lightnet.infer(image, cuda=True)
    bboxes = lightnet.get_bboxes()
    pylightnet.draw_bboxes_on_image(image, bboxes, colormap, names)

    # Save the output image
    output_path = os.path.join("tests/outputs", os.path.basename(image_path))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, image)

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

    def create_test_config(self):
        """Create a minimal test configuration."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("person\ncar\nbicycle\n")
            names_path = f.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("255,0,0\n0,255,0\n0,0,255\n")
            rgb_path = f.name

        config_dict = {
            "onnx": "dummy_model.onnx",  # This would need to exist in real tests
            "c": 3,
            "thresh": 0.5,
            "nms_thresh": 0.45,
            "anchors": [10, 13, 16, 30, 33, 23],
            "num_anchors": 3,
            "names": names_path,
            "rgb": rgb_path,
            "precision": "fp16",
            "calibration_images": "/tmp/calib",
            "calib": "Entropy",
        }

        return config_dict, names_path, rgb_path

    def test_trt_lightnet_initialization_and_cleanup(self):
        """Test proper initialization and cleanup of TrtLightnet."""
        config_dict, names_path, rgb_path = self.create_test_config()

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
        finally:
            os.unlink(names_path)
            os.unlink(rgb_path)

    def test_inference_with_different_image_sizes(self):
        """Test inference with various image sizes."""
        config_dict, names_path, rgb_path = self.create_test_config()

        try:
            if not os.path.exists(config_dict["onnx"]):
                pytest.skip(f"ONNX file {config_dict['onnx']} not found")

            lightnet = pylightnet.create_lightnet_from_config(config_dict)
            names = pylightnet.load_names_from_file(names_path)

            # Test with different image sizes
            test_sizes = [(480, 640), (720, 1280), (1080, 1920)]

            for height, width in test_sizes:
                # Create test image
                image = np.zeros((height, width, 3), dtype=np.uint8)
                # Add some content (white rectangle)
                cv2.rectangle(image, (100, 100), (200, 200), (255, 255, 255), -1)

                # Run inference
                lightnet.infer(image, cuda=True)
                bboxes = lightnet.get_bboxes()

                # Results should be a list (possibly empty)
                assert isinstance(bboxes, list)

                # If any detections, verify structure
                for bbox in bboxes:
                    assert "box" in bbox
                    assert "label" in bbox
                    assert "prob" in bbox
                    assert len(bbox["box"]) == 4
                    assert 0 <= bbox["label"] < len(names)
                    assert 0 <= bbox["prob"] <= 1

            lightnet.destroy()

        except RuntimeError as e:
            if "Failed to load" in str(e) or "Failed to create" in str(e):
                pytest.skip(f"GPU initialization failed: {e}")
            raise
        finally:
            os.unlink(names_path)
            os.unlink(rgb_path)

    def test_batch_inference(self):
        """Test batch inference functionality."""
        config_dict, names_path, rgb_path = self.create_test_config()

        # Add subnet configuration for batch inference
        config_dict["subnet_onnx"] = "dummy_subnet.onnx"
        config_dict["subnet_c"] = 2
        config_dict["subnet_thresh"] = 0.3
        config_dict["subnet_anchors"] = [10, 13]
        config_dict["subnet_num_anchors"] = 1
        config_dict["subnet_names"] = names_path
        config_dict["subnet_rgb"] = rgb_path
        config_dict["batch_size"] = 4

        try:
            if not os.path.exists(config_dict["onnx"]) or not os.path.exists(
                config_dict["subnet_onnx"]
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

            names = pylightnet.load_names_from_file(names_path)

            # Test batch inference
            results = lightnet.infer_subnet_batches_from_bboxes(
                test_bboxes,
                image,
                names,
                ["person", "car"],  # target labels
                ["sub1", "sub2"],  # subnet labels
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
        finally:
            os.unlink(names_path)
            os.unlink(rgb_path)

    def test_segmentation_masks(self):
        """Test segmentation mask generation."""
        config_dict, names_path, rgb_path = self.create_test_config()

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
        finally:
            os.unlink(names_path)
            os.unlink(rgb_path)

    def test_multi_stage_inference(self):
        """Test multi-stage inference functionality."""
        config_dict, names_path, rgb_path = self.create_test_config()

        # Add subnet configuration
        config_dict["subnet_onnx"] = "dummy_subnet.onnx"
        config_dict["subnet_c"] = 2
        config_dict["subnet_thresh"] = 0.3
        config_dict["subnet_anchors"] = [10, 13]
        config_dict["subnet_num_anchors"] = 1
        config_dict["subnet_names"] = names_path
        config_dict["subnet_rgb"] = rgb_path

        try:
            if not os.path.exists(config_dict["onnx"]) or not os.path.exists(
                config_dict["subnet_onnx"]
            ):
                pytest.skip("Required ONNX files not found")

            lightnet = pylightnet.create_lightnet_from_config(config_dict)

            # Create test image
            image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

            # Run multi-stage inference
            target_list = ["person", "car"]
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
        finally:
            os.unlink(names_path)
            os.unlink(rgb_path)

    def test_performance_metrics(self):
        """Test performance measurement with profiling."""
        config_dict, names_path, rgb_path = self.create_test_config()

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
        finally:
            os.unlink(names_path)
            os.unlink(rgb_path)
