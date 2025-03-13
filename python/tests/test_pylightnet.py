"""Test pylightnet package."""

# Avoid loading local pylightnet
from __future__ import absolute_import

import ctypes
import os

import cv2
import pylightnet
import pytest


def test_load_libs():
    """Test loading libraries."""
    # Load dependent library libcnpy.so as a global library
    libcnpy_path = os.path.join(os.path.dirname(pylightnet.__file__), "libcnpy.so")
    try:
        ctypes.CDLL(libcnpy_path, mode=ctypes.RTLD_GLOBAL)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load dependent library libcnpy.so from {libcnpy_path}"
        ) from e

    # Load the main library liblightnetinfer.so
    lib_path = os.path.join(os.path.dirname(pylightnet.__file__), "liblightnetinfer.so")
    try:
        ctypes.CDLL(lib_path)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load library liblightnetinfer.so from {lib_path}"
        ) from e


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
    if not os.path.exists(config_path):
        pytest.skip(f"Config file {config_path} not found.")
    if not os.path.exists(image_path):
        pytest.skip(f"Image file {image_path} not found.")

    # Load the model
    config_dict = pylightnet.load_config(config_path)
    onnx_path = config_dict["onnx"]
    if not os.path.exists(onnx_path):
        pytest.skip(f"ONNX file {onnx_path} not found.")

    lightnet = pylightnet.create_lightnet_from_config(config_dict)
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
