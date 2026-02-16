# Copyright 2025 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import csv
import ctypes
import json
import os
import re

import cv2
import numpy as np
import pkg_resources
from typing import List
from pathlib import Path

__all__ = ["pylightnet"]


# Define the C-compatible ModelConfigC structure
class ModelConfigC(ctypes.Structure):
    _fields_ = [
        ("model_path", ctypes.c_char * 256),
        ("num_class", ctypes.c_int),
        ("score_threshold", ctypes.c_float),
        ("anchors", ctypes.c_int * 40),
        ("anchor_elements", ctypes.c_int),
        ("num_anchors", ctypes.c_int),
        ("nms_threshold", ctypes.c_float),
        ("names", ctypes.POINTER(ctypes.c_char_p)),
        ("num_names", ctypes.c_int),
        ("detection_colormap", ctypes.POINTER(ctypes.c_int)),
        ("detection_colormap_size", ctypes.c_int),
    ]


# Define the C-compatible InferenceConfigC structure
class InferenceConfigC(ctypes.Structure):
    _fields_ = [
        ("precision", ctypes.c_char * 64),
        ("profile", ctypes.c_bool),
        ("sparse", ctypes.c_bool),
        ("dla_core_id", ctypes.c_int),
        ("use_first_layer", ctypes.c_bool),
        ("use_last_layer", ctypes.c_bool),
        ("batch_size", ctypes.c_int),
        ("scale", ctypes.c_double),
        ("calibration_images", ctypes.c_char * 512),
        ("calibration_type", ctypes.c_char * 64),
        ("max_batch_size", ctypes.c_int),
        ("min_batch_size", ctypes.c_int),
        ("optimal_batch_size", ctypes.c_int),
        ("workspace_size", ctypes.c_size_t),
    ]


# Define the C-compatible BuildConfigC structure
class BuildConfigC(ctypes.Structure):
    _fields_ = [
        ("calib_type_str", ctypes.c_char * 64),
        ("dla_core_id", ctypes.c_int),
        ("quantize_first_layer", ctypes.c_bool),
        ("quantize_last_layer", ctypes.c_bool),
        ("profile_per_layer", ctypes.c_bool),
        ("clip_value", ctypes.c_double),
        ("sparse", ctypes.c_bool),
        ("debug_tensors", (ctypes.c_char * 64) * 10),
        ("num_debug_tensors", ctypes.c_int),
    ]


# Define the C-compatible BBox and BBoxInfo structures
class BBoxC(ctypes.Structure):
    _fields_ = [
        ("x1", ctypes.c_float),
        ("y1", ctypes.c_float),
        ("x2", ctypes.c_float),
        ("y2", ctypes.c_float),
    ]


class BBoxInfoC(ctypes.Structure):
    _fields_ = [
        ("box", BBoxC),
        ("label", ctypes.c_int),
        ("classId", ctypes.c_int),
        ("prob", ctypes.c_float),
        ("isHierarchical", ctypes.c_bool),
        ("subClassId", ctypes.c_int),
        ("sin", ctypes.c_float),
        ("cos", ctypes.c_float),
        ("keypoints", ctypes.POINTER(ctypes.c_void_p)),
        ("num_keypoints", ctypes.c_int),
        ("attribute", ctypes.c_char * 256),
        ("attribute_prob", ctypes.c_float),
        ("batch_index", ctypes.c_int),  # Index in batch
    ]


class ColormapC(ctypes.Structure):
    _fields_ = [
        ("id", ctypes.c_int),
        ("name", ctypes.c_char * 50),  # Fixed-length string (50 bytes)
        ("color", ctypes.c_ubyte * 3),  # RGB as unsigned char array
        ("is_dynamic", ctypes.c_bool),
    ]


class C_CalibratedSensorInfo(ctypes.Structure):
    """
    Corresponds to the C_CalibratedSensorInfo struct defined in C++.
    This structure holds C-compatible types (pointers, sizes) for safe
    data transfer across the C/Python boundary, avoiding C++ STL containers.
    """

    _fields_ = [
        # Strings (const char*)
        ("token", ctypes.c_char_p),
        ("sensor_token", ctypes.c_char_p),
        # Vectors (double* and size_t)
        ("translation", ctypes.POINTER(ctypes.c_double)),
        ("translation_size", ctypes.c_size_t),
        ("rotation", ctypes.POINTER(ctypes.c_double)),
        ("rotation_size", ctypes.c_size_t),
        # 2D Array (Flattened double* and dimension info)
        ("camera_intrinsic", ctypes.POINTER(ctypes.c_double)),
        ("intrinsic_rows", ctypes.c_size_t),
        ("intrinsic_cols", ctypes.c_size_t),
        ("camera_distortion", ctypes.POINTER(ctypes.c_double)),
        ("distortion_size", ctypes.c_size_t),
        # Strings (const char*)
        ("name", ctypes.c_char_p),
        ("modality", ctypes.c_char_p),
        # Integers
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
    ]


class C_ImageResult(ctypes.Structure):
    """
    Corresponds to the C_ImageResult struct for transferring cv::Mat data from C++.
    """

    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_ubyte)),  # Heap-allocated image data
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("channels", ctypes.c_int),
        ("data_size", ctypes.c_size_t),
    ]


# Utility functions to load names and colormap from files
def load_names_from_file(filename):
    """Load class names from a text file."""
    with open(filename, "r") as f:
        return [line.strip() for line in f if line.strip()]


def load_colormap_from_file(filename):
    """Load colormap data from a file."""
    with open(filename, "r") as f:
        colormap = []
        for line in f:
            if line == "\n":
                continue
            values = [int(v) for v in line.strip().split(",")]
            if len(values) == 3:
                colormap.extend(values)
        return colormap


def load_segmentation_data(csv_file_path):
    """
    Load segmentation data from a CSV file and store it in a dictionary.

    :param csv_file_path: Path to the CSV file
    :return: A dictionary with 'id' as the key and segmentation properties as values
    """
    data_dict = {}
    with open(csv_file_path, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            data_dict[int(row["id"])] = {
                "name": row["name"],
                "r": int(row["r"]),
                "g": int(row["g"]),
                "b": int(row["b"]),
                "dynamic": row["dynamic"].lower()
                == "true",  # Convert string to boolean
            }
    return data_dict


def compute_iou(box1, box2):
    """Compute IoU between two boxes: [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    union = area1 + area2 - inter_area
    return inter_area / union if union > 0 else 0.0


def nms_bboxes(results, iou_thresh=0.5):
    """Apply NMS to a list of result dicts with 'box' and 'prob'."""
    if not results:
        return []

    # Sort by confidence
    sorted_results = sorted(results, key=lambda x: x["prob"], reverse=True)
    keep = []

    while sorted_results:
        best = sorted_results.pop(0)
        keep.append(best)

        sorted_results = [
            r for r in sorted_results if compute_iou(best["box"], r["box"]) < iou_thresh
        ]

    return keep


# Wrapper for TrtLightnet
class TrtLightnet:
    def __init__(
        self,
        model_config,
        inference_config,
        build_config,
        subnet_model_config=None,
        subnet_inference_config=None,
    ):
        # Load dependent library libcnpy.so as a global library
        libcnpy_path = pkg_resources.resource_filename(
            __name__, os.path.join("libcnpy.so")
        )
        try:
            ctypes.CDLL(libcnpy_path, mode=ctypes.RTLD_GLOBAL)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load dependent library libcnpy.so from {libcnpy_path}"
            ) from e

        # Load the main library liblightnetinfer.so
        lib_path = pkg_resources.resource_filename(__name__, "liblightnetinfer.so")
        try:
            self.lib = ctypes.CDLL(lib_path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load library liblightnetinfer.so from {lib_path}"
            ) from e

        # Define argument and return types
        self.lib.create_trt_lightnet.argtypes = [
            ctypes.POINTER(ModelConfigC),
            ctypes.POINTER(InferenceConfigC),
            ctypes.POINTER(BuildConfigC),
        ]
        self.lib.create_trt_lightnet.restype = ctypes.c_void_p

        self.lib.destroy_trt_lightnet.argtypes = [
            ctypes.c_void_p,
        ]
        self.lib.destroy_trt_lightnet.restype = None

        self.lib.trt_lightnet_get_input_size.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]

        self.lib.trt_lightnet_get_input_size.restype = None

        self.lib.infer_lightnet_wrapper.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_ubyte),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_bool,
        ]
        self.lib.infer_lightnet_wrapper.restype = None

        self.lib.infer_batches.argtypes = [
            ctypes.c_void_p,  # void* instance
            ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte)),  # unsigned char** imgs
            ctypes.POINTER(ctypes.c_int),  # int* heights
            ctypes.POINTER(ctypes.c_int),  # int* widths
            ctypes.POINTER(ctypes.c_int),  # int* channels
            ctypes.c_int,  # int batch_size
            ctypes.POINTER(ctypes.POINTER(BBoxInfoC)),  # BBoxInfoC** out_bboxes
            ctypes.POINTER(ctypes.c_int),  # int* out_bbox_count
        ]
        self.lib.infer_batches.restype = None

        self.lib.infer_multi_stage_lightnet_wrapper.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_ubyte),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_bool,
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.c_int,
        ]
        self.lib.infer_multi_stage_lightnet_wrapper.restype = None

        self.lib.blur_image.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_ubyte),
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.lib.blur_image.restype = None

        self.lib.get_bbox_array.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
        ]
        self.lib.get_bbox_array.restype = ctypes.POINTER(BBoxInfoC)

        self.lib.get_top_index.argtypes = [ctypes.c_void_p]
        self.lib.get_top_index.restype = ctypes.c_int

        self.lib.get_subnet_bbox_array.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
        ]
        self.lib.get_subnet_bbox_array.restype = ctypes.POINTER(BBoxInfoC)

        # Create the C++ TrtLightnet instance
        self.instance = self.lib.create_trt_lightnet(
            ctypes.byref(model_config),
            ctypes.byref(inference_config),
            ctypes.byref(build_config),
        )

        if subnet_model_config is not None:
            self.sub_instance = self.lib.create_trt_lightnet(
                ctypes.byref(subnet_model_config),
                ctypes.byref(subnet_inference_config),
                ctypes.byref(build_config),
            )

        if not self.instance:
            raise RuntimeError("Failed to create TrtLightnet instance")

        self.lib.convert_to_vec3b.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
        ]
        self.lib.convert_to_vec3b.restype = ctypes.POINTER(ctypes.c_void_p)
        self.lib.free_vec3b.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.lib.free_vec3b.restype = None

        self.lib.makeMask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        self.lib.makeMask.restype = None

        self.lib.get_masks.argtypes = [ctypes.c_void_p]
        self.lib.get_masks.restype = ctypes.POINTER(ctypes.c_void_p)

        self.lib.get_mask_count.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.lib.get_mask_count.restype = ctypes.c_size_t

        self.lib.get_mask_data.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_size_t,
        ]
        self.lib.get_mask_data.restype = ctypes.POINTER(ctypes.c_uint8)

        self.lib.get_mask_shape.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]

        self.lib.free_masks.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.lib.free_masks.restype = None

        self.lib.get_polygon_str.argtypes = [
            ctypes.c_void_p,  # instance (pointer to shared_ptr)
            ctypes.c_int,
            ctypes.c_int,  # width, height
            ctypes.POINTER(ColormapC),
            ctypes.c_size_t,  # colormap array and length
            ctypes.c_char_p,  # image_name
        ]
        self.lib.get_polygon_str.restype = ctypes.c_char_p  # Returns a string

        self.lib.makeEntropy.argtypes = [ctypes.c_void_p]
        self.lib.makeEntropy.restype = None

        self.lib.get_entropy_maps.argtypes = [ctypes.c_void_p]
        self.lib.get_entropy_maps.restype = ctypes.POINTER(ctypes.c_void_p)

        self.lib.get_entropy_count.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.lib.get_entropy_count.restype = ctypes.c_size_t

        self.lib.get_entropy_data.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_size_t,
        ]
        self.lib.get_entropy_data.restype = ctypes.POINTER(ctypes.c_uint8)

        self.lib.get_entropy_shape.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]

        self.lib.free_entropy_maps.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.lib.free_entropy_maps.restype = None

        self.lib.get_entropies.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        self.lib.get_entropies.restype = None

        self.lib.free_entropies.argtypes = [ctypes.POINTER(ctypes.c_float)]
        self.lib.free_entropies.restype = None

        # Set argument types (argtypes) and return type (restype) for the C++ getter
        # Signature: get_calibrated_info_for_python(const char*, const char*) -> C_CalibratedSensorInfo
        self.lib.get_calibrated_info_for_python.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
        ]
        self.lib.get_calibrated_info_for_python.restype = C_CalibratedSensorInfo

        # Set signature for the C++ memory deallocation function
        # Signature: free_calibrated_info(C_CalibratedSensorInfo) -> void
        self.lib.free_calibrated_info.argtypes = [C_CalibratedSensorInfo]
        self.lib.free_calibrated_info.restype = None

        self.lib.make_range_image_for_python.argtypes = [
            ctypes.c_char_p,
            C_CalibratedSensorInfo,
            ctypes.c_float,
        ]
        self.lib.make_range_image_for_python.restype = C_ImageResult

        # free_image_result(C_ImageResult) -> void
        self.lib.free_image_result.argtypes = [C_ImageResult]
        self.lib.free_image_result.restype = None

    def get_input_size(self):
        batch = ctypes.c_int()
        chan = ctypes.c_int()
        height = ctypes.c_int()
        width = ctypes.c_int()
        self.lib.trt_lightnet_get_input_size(
            self.instance,
            ctypes.byref(batch),
            ctypes.byref(chan),
            ctypes.byref(height),
            ctypes.byref(width),
        )
        return batch.value, chan.value, height.value, width.value

    def infer(self, image, cuda=False):
        if image.dtype != np.uint8:
            raise ValueError("Image must be of type np.uint8")
        height, width, _ = image.shape
        img_data = image.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
        self.lib.infer_lightnet_wrapper(self.instance, img_data, width, height, cuda)

    def destroy(self):
        self.lib.destroy_trt_lightnet(self.instance)
        if hasattr(self, "sub_instance"):
            self.lib.destroy_trt_lightnet(self.sub_instance)

    def infer_subnet_batches_from_bboxes(
        self,
        bboxes,
        image,
        all_labels,
        target_labels,
        subnet_labels,
        batch_size,
        min_crop_size,
        debug=False,
    ):
        """Filter, classify, and annotate bounding boxes on the input image."""

        valid_bboxes = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox["box"]
            width, height = x2 - x1, y2 - y1
            if width < min_crop_size or height < min_crop_size:
                continue

            label_id = bbox["label"]
            class_name = (
                all_labels[label_id] if label_id < len(all_labels) else "Unknown"
            )
            if class_name in target_labels:
                valid_bboxes.append(bbox)

        results = []

        for i in range(0, len(valid_bboxes), batch_size):
            current_batch = valid_bboxes[i : i + batch_size]

            cropped_imgs = [
                image[int(y1) : int(y2), int(x1) : int(x2)].copy()
                for x1, y1, x2, y2 in (bbox["box"] for bbox in current_batch)
            ]

            num = len(cropped_imgs)
            img_ptrs = (ctypes.POINTER(ctypes.c_ubyte) * num)()
            heights = (ctypes.c_int * num)()
            widths = (ctypes.c_int * num)()
            channels = (ctypes.c_int * num)()

            for idx, img in enumerate(cropped_imgs):
                img = np.ascontiguousarray(img, dtype=np.uint8)
                img_ptrs[idx] = img.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
                heights[idx] = img.shape[0]
                widths[idx] = img.shape[1]
                channels[idx] = img.shape[2] if img.ndim == 3 else 1

            bbox_ptr = ctypes.POINTER(BBoxInfoC)()
            bbox_count = ctypes.c_int()

            self.lib.infer_batches(
                self.sub_instance,
                img_ptrs,
                heights,
                widths,
                channels,
                num,
                ctypes.byref(bbox_ptr),
                ctypes.byref(bbox_count),
            )

            for j in range(bbox_count.value):
                b = bbox_ptr[j]
                orig_box = current_batch[b.batch_index]["box"]
                x_offset, y_offset = orig_box[0], orig_box[1]

                result = {
                    "box": [
                        b.box.x1 + x_offset,
                        b.box.y1 + y_offset,
                        b.box.x2 + x_offset,
                        b.box.y2 + y_offset,
                    ],
                    "label": b.label,
                    "classId": b.classId,
                    "prob": b.prob,
                    "attribute": b.attribute.decode("utf-8", errors="ignore"),
                    "attribute_prob": b.attribute_prob,
                }
                results.append(result)

            # Free memory
            libc = ctypes.CDLL("libc.so.6")
            libc.free.argtypes = [ctypes.c_void_p]
            libc.free(bbox_ptr)

        filtered_results = nms_bboxes(results, iou_thresh=0.45)
        return filtered_results

    def infer_multi_stage(self, image, target_list, cuda=False):
        if image.dtype != np.uint8:
            raise ValueError("Image must be of type np.uint8")
        height, width, _ = image.shape
        img_data = image.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
        c_array = (ctypes.c_char_p * len(target_list))(
            *[s.encode("utf-8") for s in target_list]
        )
        self.lib.infer_multi_stage_lightnet_wrapper(
            self.instance,
            self.sub_instance,
            img_data,
            width,
            height,
            cuda,
            c_array,
            len(target_list),
        )

    def classifer_from_bboxes(
        self, image, bboxes, names, target, sub_names, cuda=False
    ):
        count = 0
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox["box"]
            label = bbox["label"]
            class_name = names[label] if label < len(names) else "Unknown"
            if class_name != target:
                continue
            cropped = image[int(y1) : int(y2), int(x1) : int(x2)].copy()
            height, width, _ = cropped.shape
            img_data = cropped.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
            self.lib.infer_lightnet_wrapper(
                self.instance, img_data, width, height, False
            )
            top_index = self.lib.get_top_index(self.instance)

            bbox["sub_name"] = sub_names[top_index]
            count = count + 1

    def blur_image(self, image):
        if image.dtype != np.uint8:
            raise ValueError("Image must be of type np.uint8")
        height, width, _ = image.shape
        img_data = image.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
        self.lib.blur_image(self.instance, self.sub_instance, img_data, width, height)

    def get_bboxes(self):
        size = ctypes.c_int()
        bbox_array = self.lib.get_bbox_array(self.instance, ctypes.byref(size))
        bboxes = []
        for i in range(size.value):
            bbox = bbox_array[i]
            bboxes.append(
                {
                    "box": (bbox.box.x1, bbox.box.y1, bbox.box.x2, bbox.box.y2),
                    "label": bbox.label,
                    "classId": bbox.classId,
                    "prob": bbox.prob,
                    "subClassId": bbox.subClassId,
                    "sin": bbox.sin,
                    "cos": bbox.cos,
                    "id": i,
                }
            )
        return bboxes

    def get_subnet_bboxes(self):
        size = ctypes.c_int()
        bbox_array = self.lib.get_subnet_bbox_array(self.instance, ctypes.byref(size))
        bboxes = []
        for i in range(size.value):
            bbox = bbox_array[i]
            bboxes.append(
                {
                    "box": (bbox.box.x1, bbox.box.y1, bbox.box.x2, bbox.box.y2),
                    "label": bbox.label,
                    "classId": bbox.classId,
                    "prob": bbox.prob,
                    "subClassId": bbox.subClassId,
                    "sin": bbox.sin,
                    "cos": bbox.cos,
                }
            )

        return bboxes

    def make_mask(self, argmax2bgr_ptr):
        """
        Call the C++ function TrtLightnet::makeMask with the given std::vector<cv::Vec3b>.

        :param argmax2bgr_ptr: Pointer to std::vector<cv::Vec3b>
        """
        self.lib.makeMask(self.instance, argmax2bgr_ptr)

    def segmentation_to_argmax2bgr(self, segmentation_data):
        """
        Convert segmentation data to a C++ std::vector<cv::Vec3b> via ctypes.

        :param segmentation_data: Dictionary with segmentation IDs and RGB values.
        :return: Pointer to std::vector<cv::Vec3b> in C++.
        """
        # Extract RGB values from segmentation_data
        rgb_list = []
        for key in sorted(segmentation_data.keys()):  # Ensure order consistency
            rgb_list.extend(
                [
                    segmentation_data[key]["b"],  # OpenCV uses BGR format
                    segmentation_data[key]["g"],
                    segmentation_data[key]["r"],
                ]
            )

        # Convert list to a NumPy array (uint8)
        rgb_array = np.array(rgb_list, dtype=np.uint8)

        # Call C++ function to convert to std::vector<cv::Vec3b>
        self.argmax2bgr_ptr = self.lib.convert_to_vec3b(
            rgb_array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), rgb_array.size
        )

        return self.argmax2bgr_ptr

    def free_argmax2bgr(self):
        self.lib.free_vec3b(self.argmax2bgr_ptr)

    def get_masks_from_cpp(self):
        """
        Calls TrtLightnet::getMask() and retrieves masks as NumPy arrays.
        """
        # Get masks from C++
        masks_ptr = self.lib.get_masks(self.instance)

        # Get the number of masks
        mask_count = self.lib.get_mask_count(masks_ptr)

        masks = []

        for i in range(mask_count):
            rows = ctypes.c_int()
            cols = ctypes.c_int()
            channels = ctypes.c_int()

            # Get shape of the mask
            self.lib.get_mask_shape(
                masks_ptr,
                i,
                ctypes.byref(rows),
                ctypes.byref(cols),
                ctypes.byref(channels),
            )
            rows, cols, channels = rows.value, cols.value, channels.value

            # Get mask data
            data_ptr = self.lib.get_mask_data(masks_ptr, i)

            if data_ptr:
                # Convert to NumPy array
                array = np.ctypeslib.as_array(data_ptr, shape=(rows, cols, channels))
                masks.append(array.copy())  # Copy to avoid memory issues

        # Free the allocated memory in C++
        self.lib.free_masks(masks_ptr)

        return masks

    def get_polygon_str_from_masks(self, width, height, data_dict, filename):
        colormap_list = []

        for key in sorted(data_dict.keys()):  # Ensure order consistency
            entry = data_dict[key]
            colormap = ColormapC(
                id=key,
                name=entry["name"].encode("utf-8"),  # Encode to bytes
                color=(ctypes.c_ubyte * 3)(entry["r"], entry["g"], entry["b"]),
                is_dynamic=entry["dynamic"],
            )
            colormap_list.append(colormap)

        # Convert list to ctypes array
        colormap_array = (ColormapC * len(colormap_list))(*colormap_list)
        c_string = filename.encode("utf-8")
        # Call the C++ function
        result_ptr = self.lib.get_polygon_str(
            self.instance, width, height, colormap_array, len(colormap_list), c_string
        )
        result_str = result_ptr.decode("utf-8")
        image_annotations = json.loads(result_str)
        self.make_entropy()
        entropies = self.get_entropies()
        if len(entropies) > 0:
            image_annotations["uncertainty"] = entropies[0].tolist()
        return image_annotations

    def make_entropy(self):
        """
        Invoke the C++ method TrtLightnet::calcEntropyFromSoftmax via ctypes.
        """
        self.lib.makeEntropy(self.instance)

    def get_entropies(self):
        """
        Retrieve entropy values (std::vector<std::vector<float>>) from C++ and return as a NumPy 2D array.

        Returns:
        np.ndarray: A 2D matrix of shape (outer_size, inner_size) containing entropy values.
        Raises:
        RuntimeError: If the C++ call fails or returns a null pointer.
        """
        data_ptr = ctypes.POINTER(ctypes.c_float)()
        outer = ctypes.c_int()
        inner = ctypes.c_int()

        self.lib.get_entropies(
            self.instance,
            ctypes.byref(data_ptr),
            ctypes.byref(outer),
            ctypes.byref(inner),
        )

        outer_size = outer.value
        inner_size = inner.value

        if not bool(data_ptr):
            raise RuntimeError("Failed to get entropy data from C++.")

        # Convert C array to NumPy array
        flat_array = np.ctypeslib.as_array(data_ptr, shape=(outer_size * inner_size,))
        matrix = flat_array.reshape(
            (outer_size, inner_size)
        ).copy()  # Deep copy to own the data

        # Free the memory allocated in C++
        self.lib.free_entropies(data_ptr)

        return matrix

    def get_entropy_maps_from_cpp(self):
        """
        Retrieve entropy visualization maps (std::vector<cv::Mat>) from C++.

        Returns:
        List[np.ndarray]: List of HxWxC NumPy arrays (typically C=3 for BGR images).
        """
        entropy_maps_ptr = self.lib.get_entropy_maps(self.instance)
        entropy_count = self.lib.get_entropy_count(entropy_maps_ptr)

        entropy_maps = []

        for i in range(entropy_count):
            rows = ctypes.c_int()
            cols = ctypes.c_int()
            channels = ctypes.c_int()

            # Get shape info of each cv::Mat
            self.lib.get_entropy_shape(
                entropy_maps_ptr,
                i,
                ctypes.byref(rows),
                ctypes.byref(cols),
                ctypes.byref(channels),
            )

            r, c, ch = rows.value, cols.value, channels.value

            # Get raw image data pointer
            data_ptr = self.lib.get_entropy_data(entropy_maps_ptr, i)

            if data_ptr:
                np_image = np.ctypeslib.as_array(data_ptr, shape=(r, c, ch))
                entropy_maps.append(
                    np_image.copy()
                )  # Copy is necessary to avoid dangling pointers

        # Free the vector<cv::Mat> allocated in C++
        self.lib.free_entropy_maps(entropy_maps_ptr)

        return entropy_maps

    def get_calibrated_info(self, calib_path: str, camera_name: str):
        """
        Calls the C++ function to retrieve calibrated sensor info, converts the result
        into a Python dictionary, and frees the C++ allocated memory.

        Args:
        calib_path: Path to the directory containing calibration data.
        camera_name: Name of the target camera.

        Returns:
        A dictionary containing the calibrated sensor information.
        """
        if not self.lib:
            raise RuntimeError("C library is not loaded. Cannot proceed.")

        # Encode Python strings to bytes for C compatibility
        # path_bytes = calib_path.encode('utf-8')
        path_str = str(calib_path)
        path_bytes = path_str.encode("utf-8")
        name_bytes = camera_name.encode("utf-8")

        # Call the C++ function to get the C-compatible struct
        c_info = self.lib.get_calibrated_info_for_python(path_bytes, name_bytes)

        result = {}
        try:
            # --- Data Conversion ---

            # 1. Strings (c_char_p)
            result["token"] = c_info.token.decode("utf-8") if c_info.token else ""
            result["sensor_token"] = (
                c_info.sensor_token.decode("utf-8") if c_info.sensor_token else ""
            )
            result["name"] = c_info.name.decode("utf-8") if c_info.name else ""
            result["modality"] = (
                c_info.modality.decode("utf-8") if c_info.modality else ""
            )

            # 2. 1D Arrays (POINTER(c_double))
            result["translation"] = list(c_info.translation[: c_info.translation_size])
            result["rotation"] = list(c_info.rotation[: c_info.rotation_size])
            result["camera_distortion"] = list(
                c_info.camera_distortion[: c_info.distortion_size]
            )

            # 3. 2D Array (Intrinsic Matrix)
            intrinsic_matrix: List[List[float]] = []
            if (
                c_info.camera_intrinsic
                and c_info.intrinsic_rows > 0
                and c_info.intrinsic_cols > 0
            ):
                total_size = c_info.intrinsic_rows * c_info.intrinsic_cols
                intrinsic_flat = list(c_info.camera_intrinsic[:total_size])

                # Reshape the flattened list into a 2D matrix
                for i in range(c_info.intrinsic_rows):
                    start_index = i * c_info.intrinsic_cols
                    end_index = (i + 1) * c_info.intrinsic_cols
                    intrinsic_matrix.append(intrinsic_flat[start_index:end_index])

            result["camera_intrinsic"] = intrinsic_matrix

            # 4. Primitive types
            result["width"] = c_info.width
            result["height"] = c_info.height

        except Exception as e:
            # エラー時はここでc_infoのメモリを解放すべき
            self.lib.free_calibrated_info(c_info)
            raise e

        return result, c_info

    ## 🎯 Range Image Generation Wrapper

    def make_range_image(
        self,
        file_path: Path,
        target_calib_info: C_CalibratedSensorInfo,
        max_distance: float,
    ) -> np.ndarray:
        """
        Calls the C++ function Pcd2image::makeRangeImageFromCalibration via ctypes.

        This function securely converts the Path object, passes the calibration struct,
        retrieves the image data (cv::Mat content), converts it to a NumPy array,
        and frees the memory allocated by the C++ layer.

        Args:
        file_path: Path object pointing to the input point cloud file.
        target_calib_info: C_CalibratedSensorInfo struct containing calibration data.
        max_distance: Maximum distance for range image normalization (float).

        Returns:
        A NumPy array representing the range image (H x W x C, uint8), or an empty array on failure.
        """
        if not self.lib:
            print("Error: C library is not loaded. Returning empty array.")
            return np.array([], dtype=np.uint8)

        # 1. Convert Path object to a C-compatible byte string (const char*)
        # Path -> str -> bytes (utf-8 encoded)
        path_bytes = str(file_path).encode("utf-8")

        # 2. Call the C++ wrapper function to get the C_ImageResult struct
        c_result = self.lib.make_range_image_for_python(
            path_bytes, target_calib_info, max_distance
        )

        image_data = np.array([], dtype=np.uint8)

        try:
            if c_result.data and c_result.data_size > 0:
                width = c_result.width
                height = c_result.height
                channels = c_result.channels
                data_size = c_result.data_size

                if (
                    height > 0
                    and width > 0
                    and channels > 0
                    and data_size == width * height * channels
                ):
                    # 3. Create a NumPy array view of the C-allocated memory
                    # Create the NumPy array pointing to the memory block
                    np_array_view = np.ctypeslib.as_array(
                        c_result.data, shape=(data_size,)
                    )

                    # 4. Reshape the 1D view into the final 2D/3D image shape (H x W x C) and copy.
                    # The .copy() is crucial: it moves the data from the C++ heap to Python's memory,
                    # allowing us to safely free the C++ memory in the 'finally' block.
                    image_data = np_array_view.reshape(height, width, channels).copy()

                else:
                    print(
                        f"Warning: Invalid image dimensions received: H={height}, W={width}, C={channels}, Size={data_size}"
                    )

        finally:
            # 🔥 ESSENTIAL: Free the memory allocated by the C++ layer's malloc/new
            if self.lib and c_result.data:
                self.lib.free_image_result(c_result)

        return image_data


def load_config(file_path):
    """Load configuration from a file and return it as a dictionary."""
    config_dict = {}
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue  # Ignore comments and empty lines

            key, value = (
                line.lstrip("--").split("=", 1)
                if "=" in line
                else (line.lstrip("--"), None)
            )

            # Convert values to appropriate types
            if value is not None:
                if value.isdigit():
                    value = int(value)
                elif value.replace(".", "", 1).isdigit():
                    value = float(value)
                elif value.lower() in ["true", "false"]:
                    value = value.lower() == "true"
                elif "," in value:
                    value = re.sub(r"\s+", "", value)
                    value = [
                        int(v)
                        if v.isdigit()
                        else float(v)
                        if v.replace(".", "", 1).isdigit()
                        else v
                        for v in value.split(",")
                    ]

            config_dict[key] = value

    return config_dict


def parse_model_config(config_dict):
    """Parse model configuration and store it in a ModelConfigC object."""
    model_config = ModelConfigC()
    model_config.model_path = config_dict["onnx"].encode("utf-8")
    model_config.num_class = int(config_dict["c"])
    model_config.score_threshold = float(config_dict["thresh"])

    # Convert anchors to a ctypes array\
    """
    if not isinstance(config_dict["anchors"], list):
        raise ValueError("anchors should be a list of integers")
    """
    if "anchors" in config_dict:
        model_config.anchors = (ctypes.c_int * 40)(*config_dict["anchors"])
        model_config.anchor_elements = len(config_dict["anchors"])
        model_config.num_anchors = int(config_dict["num_anchors"])

    # Set NMS threshold (default: 0.45)
    model_config.nms_threshold = float(config_dict.get("nms_thresh", 0.45))

    # Convert class names to ctypes array
    names = load_names_from_file(config_dict["names"])
    c_names = (ctypes.c_char_p * len(names))(*[name.encode("utf-8") for name in names])
    model_config.names = ctypes.cast(c_names, ctypes.POINTER(ctypes.c_char_p))
    model_config.num_names = len(names)

    # Convert colormap to ctypes array
    detection_colormap = load_colormap_from_file(config_dict["rgb"])
    c_detection_colormap = (ctypes.c_int * len(detection_colormap))(*detection_colormap)
    model_config.detection_colormap = ctypes.cast(
        c_detection_colormap, ctypes.POINTER(ctypes.c_int)
    )
    model_config.detection_colormap_size = len(detection_colormap)

    return model_config


def parse_inference_config(config_dict):
    """Parse inference configuration and store it in an InferenceConfigC object."""
    inference_config = InferenceConfigC()
    inference_config.precision = config_dict["precision"].encode("utf-8")
    inference_config.profile = False
    if "sparse" in config_dict:
        inference_config.sparse = bool(config_dict["sparse"])
    inference_config.dla_core_id = -1
    inference_config.use_first_layer = False
    inference_config.use_last_layer = False
    inference_config.batch_size = 1
    inference_config.scale = 1.0
    inference_config.calibration_images = config_dict["calibration_images"].encode(
        "utf-8"
    )
    inference_config.calibration_type = config_dict["calib"].encode("utf-8")

    inference_config.max_batch_size = 1
    inference_config.min_batch_size = 1
    inference_config.optimal_batch_size = 1
    inference_config.workspace_size = 1073741824  # 1 GB

    return inference_config


def parse_build_config(config_dict):
    """Parse build configuration and store it in a BuildConfigC object."""
    build_config = BuildConfigC()
    build_config.calib_type_str = b"Entropy"
    build_config.dla_core_id = -1
    build_config.quantize_first_layer = False
    build_config.quantize_last_layer = False
    build_config.profile_per_layer = True
    build_config.clip_value = -1
    if "sparse" in config_dict:
        build_config.sparse = bool(config_dict["sparse"])

    # Set debug tensors
    debug_tensor_names = [b"tensor1", b"tensor2"]
    for i, name in enumerate(debug_tensor_names):
        if i < len(build_config.debug_tensors):  # Ensure not exceeding the array size
            build_config.debug_tensors[i] = ctypes.create_string_buffer(name, 64)

    build_config.num_debug_tensors = len(debug_tensor_names)

    return build_config


# Function to create a TrtLightnet instance from a configuration dictionary
def create_lightnet_from_config(config_dict):
    """
    Initialize a TrtLightnet instance using the provided configuration dictionary.

    :param config_dict: Dictionary containing configuration settings
    :return: TrtLightnet instance
    """
    model_config = parse_model_config(config_dict)
    inference_config = parse_inference_config(config_dict)
    build_config = parse_build_config(config_dict)
    if "subnet_onnx" in config_dict:
        # Parse subnet like TLR and anonymizer
        subnet_model_config = parse_subnet_model_config(config_dict)
        subnet_inference_config = parse_inference_config(config_dict)
        if "batch_size" in config_dict:
            batch_size = config_dict["batch_size"]
            subnet_inference_config.max_batch_size = batch_size
            subnet_inference_config.optimal_batch_size = (int)(batch_size / 2)
            subnet_inference_config.min_batch_size = 1
        return TrtLightnet(
            model_config,
            inference_config,
            build_config,
            subnet_model_config,
            subnet_inference_config,
        )
    else:
        return TrtLightnet(model_config, inference_config, build_config)


def parse_subnet_model_config(config_dict):
    """Parse model configuration and store it in a ModelConfigC object."""
    model_config = ModelConfigC()
    model_config.model_path = config_dict["subnet_onnx"].encode("utf-8")
    model_config.num_class = int(config_dict["subnet_c"])
    model_config.score_threshold = float(config_dict["subnet_thresh"])

    # Convert anchors to a ctypes array\
    """
    if not isinstance(config_dict["anchors"], list):
        raise ValueError("anchors should be a list of integers")
    """
    if "anchors" in config_dict:
        model_config.anchors = (ctypes.c_int * 40)(*config_dict["subnet_anchors"])
        model_config.anchor_elements = len(config_dict["subnet_anchors"])
        model_config.num_anchors = int(config_dict["subnet_num_anchors"])

    # Set NMS threshold (default: 0.25)
    model_config.nms_threshold = float(config_dict.get("subnet_nms_thresh", 0.25))

    # Convert class names to ctypes array
    names = load_names_from_file(config_dict["subnet_names"])
    c_names = (ctypes.c_char_p * len(names))(*[name.encode("utf-8") for name in names])
    model_config.names = ctypes.cast(c_names, ctypes.POINTER(ctypes.c_char_p))
    model_config.num_names = len(names)

    # Convert colormap to ctypes array
    detection_colormap = load_colormap_from_file(config_dict["subnet_rgb"])
    c_detection_colormap = (ctypes.c_int * len(detection_colormap))(*detection_colormap)
    model_config.detection_colormap = ctypes.cast(
        c_detection_colormap, ctypes.POINTER(ctypes.c_int)
    )
    model_config.detection_colormap_size = len(detection_colormap)

    return model_config


def merge_bbox(bboxes1, bboxes2, names1, names2):
    """
    Merge bboxes2 into bboxes1, matching label names via names1/names2 and applying NMS.

    Args:
        bboxes1 (list): List of bbox dicts (target list to be updated)
        bboxes2 (list): List of bbox dicts to be merged
        names1 (list): List of class names for bboxes1 (index = label)
        names2 (list): List of class names for bboxes2 (index = label)

    Returns:
        list: Merged and NMS-applied bbox list
    """
    for bb2 in bboxes2:
        label = bb2["label"]
        name = names2[label]

        # Try to find matching name in names1
        try:
            index = names1.index(name)
            new_bb = copy.deepcopy(bb2)
            new_bb["label"] = index
            bboxes1.append(new_bb)
        except ValueError:
            continue  # name not found in names1

    ret = nms_bboxes(bboxes1, iou_thresh=0.45)
    return ret


def blur_sensitive_regions(image, bboxes, label_names, blur_kernel=(21, 21)):
    """
    Apply blur to regions labeled as LICENSE_PLATE or HUMAN_HEAD.

    Args:
        image (np.ndarray): The image to apply blur on (will be modified in-place).
        bboxes (list): List of dicts with "box" and "label".
        label_names (list): List of label names (label index to string).
        blur_kernel (tuple): Kernel size for blurring.

    Returns:
        np.ndarray: The image with blurred regions.
    """
    blurred_image = image.copy()

    for bbox in bboxes:
        label_id = bbox["label"]
        if label_id >= len(label_names):
            continue

        label_name = label_names[label_id]
        if label_name in ["LICENSE_PLATE", "HUMAN_HEAD"]:
            x1, y1, x2, y2 = map(int, bbox["box"])
            roi = blurred_image[y1:y2, x1:x2]
            if roi.size > 0:
                blurred_roi = cv2.GaussianBlur(roi, blur_kernel, 0)
                blurred_image[y1:y2, x1:x2] = blurred_roi

    return blurred_image
