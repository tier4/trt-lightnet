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

import pkg_resources
import os
import cv2
import numpy as np
import ctypes
import csv
import json
import re
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
    ]

class ColormapC(ctypes.Structure):
    _fields_ = [
        ("id", ctypes.c_int),
        ("name", ctypes.c_char * 50),  # Fixed-length string (50 bytes)
        ("color", ctypes.c_ubyte * 3),  # RGB as unsigned char array
        ("is_dynamic", ctypes.c_bool)
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
            if line == "\n" :
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
                "dynamic": row["dynamic"].lower() == "true"  # Convert string to boolean
            }
    return data_dict
        
# Wrapper for TrtLightnet
class TrtLightnet:
    def __init__(self, model_config, inference_config, build_config, subnet_model_config=None, subnet_inference_config=None):
        # Load dependent library libcnpy.so as a global library
        libcnpy_path = pkg_resources.resource_filename(__name__, os.path.join("libcnpy.so"))
        try:
            ctypes.CDLL(libcnpy_path, mode=ctypes.RTLD_GLOBAL)
        except Exception as e:
            raise RuntimeError(f"Failed to load dependent library libcnpy.so from {libcnpy_path}") from e

        # Load the main library liblightnetinfer.so
        lib_path = pkg_resources.resource_filename(__name__, "liblightnetinfer.so")
        try:
            self.lib = ctypes.CDLL(lib_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load library liblightnetinfer.so from {lib_path}") from e

        # Define argument and return types
        self.lib.create_trt_lightnet.argtypes = [
            ctypes.POINTER(ModelConfigC),
            ctypes.POINTER(InferenceConfigC),
            ctypes.POINTER(BuildConfigC),
        ]
        self.lib.create_trt_lightnet.restype = ctypes.c_void_p

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
        
        self.lib.get_bbox_array.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
        self.lib.get_bbox_array.restype = ctypes.POINTER(BBoxInfoC)

        self.lib.get_top_index.argtypes = [ctypes.c_void_p]
        self.lib.get_top_index.restype = ctypes.c_int    

        self.lib.get_subnet_bbox_array.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
        self.lib.get_subnet_bbox_array.restype = ctypes.POINTER(BBoxInfoC)        
                
        # Create the C++ TrtLightnet instance
        self.instance = self.lib.create_trt_lightnet(
            ctypes.byref(model_config),
            ctypes.byref(inference_config),
            ctypes.byref(build_config),
        )

        if subnet_model_config != None :
            self.sub_instance = self.lib.create_trt_lightnet(
                ctypes.byref(subnet_model_config),
                ctypes.byref(subnet_inference_config),
                ctypes.byref(build_config),
            )
        
        if not self.instance:
            raise RuntimeError("Failed to create TrtLightnet instance")

        self.lib.convert_to_vec3b.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
        self.lib.convert_to_vec3b.restype = ctypes.POINTER(ctypes.c_void_p)
        self.lib.free_vec3b.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.lib.free_vec3b.restype = None

        self.lib.makeMask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        self.lib.makeMask.restype = None

        self.lib.get_masks.argtypes = [ctypes.c_void_p]
        self.lib.get_masks.restype = ctypes.POINTER(ctypes.c_void_p)

        self.lib.get_mask_count.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.lib.get_mask_count.restype = ctypes.c_size_t

        self.lib.get_mask_data.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.lib.get_mask_data.restype = ctypes.POINTER(ctypes.c_uint8)

        self.lib.get_mask_shape.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]

        self.lib.free_masks.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.lib.free_masks.restype = None

        self.lib.get_polygon_str.argtypes = [
            ctypes.c_void_p,  # instance (pointer to shared_ptr)
            ctypes.c_int, ctypes.c_int,  # width, height
            ctypes.POINTER(ColormapC), ctypes.c_size_t,  # colormap array and length
            ctypes.c_char_p  # image_name
        ]
        self.lib.get_polygon_str.restype = ctypes.c_char_p  # Returns a string

        

    def get_input_size(self):
        batch = ctypes.c_int()
        chan = ctypes.c_int()
        height = ctypes.c_int()
        width = ctypes.c_int()
        self.lib.trt_lightnet_get_input_size(
            self.instance, ctypes.byref(batch), ctypes.byref(chan), ctypes.byref(height), ctypes.byref(width)
        )
        return batch.value, chan.value, height.value, width.value

    def infer(self, image, cuda=False):
        if image.dtype != np.uint8:
            raise ValueError("Image must be of type np.uint8")
        height, width, _ = image.shape
        img_data = image.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
        self.lib.infer_lightnet_wrapper(self.instance, img_data, width, height, cuda)

    def infer_multi_stage(self, image, target_list, cuda=False):
        if image.dtype != np.uint8:
            raise ValueError("Image must be of type np.uint8")
        height, width, _ = image.shape
        img_data = image.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
        c_array = (ctypes.c_char_p * len(target_list))(*[s.encode('utf-8') for s in target_list])
        self.lib.infer_multi_stage_lightnet_wrapper(self.instance, self.sub_instance, img_data, width, height, cuda, c_array, len(target_list))        

    def classifer_from_bboxes(self, image, bboxes, names, target, sub_names, cuda=False) :
        count = 0
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox["box"]
            label = bbox["label"]
            class_name = names[label] if label < len(names) else "Unknown"
            if class_name != target :
                continue
            cropped = image[int(y1):int(y2), int(x1):int(x2)].copy()
            height, width, _ = cropped.shape
            img_data = cropped.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
            self.lib.infer_lightnet_wrapper(self.instance, img_data, width, height, False)
            top_index = self.lib.get_top_index(self.instance)

            bbox["sub_name"] = sub_names[top_index]
            count = count+1
            
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
            bboxes.append({
                "box": (bbox.box.x1, bbox.box.y1, bbox.box.x2, bbox.box.y2),
                "label": bbox.label,
                "classId": bbox.classId,
                "prob": bbox.prob,
                "subClassId" : bbox.subClassId,
                "sin" : bbox.sin,
                "cos" : bbox.cos,
                "id" : i,
            })
        return bboxes

    def get_subnet_bboxes(self):
        size = ctypes.c_int()
        bbox_array = self.lib.get_subnet_bbox_array(self.instance, ctypes.byref(size))
        bboxes = []
        for i in range(size.value):
            bbox = bbox_array[i]
            bboxes.append({
                "box": (bbox.box.x1, bbox.box.y1, bbox.box.x2, bbox.box.y2),
                "label": bbox.label,
                "classId": bbox.classId,
                "prob": bbox.prob,
                "subClassId" : bbox.subClassId,
                "sin" : bbox.sin,
                "cos" : bbox.cos,                
            })

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
            rgb_list.extend([
                segmentation_data[key]["b"],  # OpenCV uses BGR format
                segmentation_data[key]["g"],
                segmentation_data[key]["r"]
            ])

        # Convert list to a NumPy array (uint8)
        rgb_array = np.array(rgb_list, dtype=np.uint8)

        # Call C++ function to convert to std::vector<cv::Vec3b>
        self.argmax2bgr_ptr = self.lib.convert_to_vec3b(rgb_array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), rgb_array.size)

        return self.argmax2bgr_ptr

    def free_argmax2bgr(self) :
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
            self.lib.get_mask_shape(masks_ptr, i, ctypes.byref(rows), ctypes.byref(cols), ctypes.byref(channels))
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
                is_dynamic=entry["dynamic"]
            )
            colormap_list.append(colormap)

        # Convert list to ctypes array
        colormap_array = (ColormapC * len(colormap_list))(*colormap_list)
        c_string = filename.encode('utf-8')
        # Call the C++ function
        result_ptr = self.lib.get_polygon_str(self.instance, width, height, colormap_array, len(colormap_list), c_string)
        result_str = result_ptr.decode("utf-8")
        image_annotations = json.loads(result_str)        
        return image_annotations

# Function to draw bounding boxes
def draw_bboxes_on_image(image, bboxes, colormap, names, filled=False):
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox["box"]
        label = bbox["label"]
        class_name = names[label] if label < len(names) else "Unknown"
        color = (colormap[label * 3 + 2], colormap[label * 3 + 1], colormap[label * 3 + 0])
        if filled == True :
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, -1)
        else :
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(image, f"{class_name} ({bbox['prob']:.2f})", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        if "attribute_prob" in bbox and bbox["attribute_prob"] > 0.1 :
            attr = bbox['attribute']
            attr_prob = bbox["attribute_prob"]
            cv2.putText(image, f"--> {attr} ({attr_prob:.2f})", (int(x1), int(y1) + 20), cv2.FONT_ITALIC, 0.5, color, 2)
        if "sub_name" in bbox:
            cv2.putText(image, f"--> {bbox['sub_name']}", (int(x1), int(y1) + 20), cv2.FONT_ITALIC, 0.5, color, 2)            



def load_config(file_path):
    """Load configuration from a file and return it as a dictionary."""
    config_dict = {}
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue  # Ignore comments and empty lines

            key, value = line.lstrip("--").split("=", 1) if "=" in line else (line.lstrip("--"), None)

            # Convert values to appropriate types
            if value is not None:
                if value.isdigit():
                    value = int(value)
                elif value.replace('.', '', 1).isdigit():
                    value = float(value)
                elif value.lower() in ["true", "false"]:
                    value = value.lower() == "true"
                elif "," in value:
                    value = re.sub(r"\s+", "", value)
                    value = [int(v) if v.isdigit() else float(v) if v.replace('.', '', 1).isdigit() else v for v in value.split(",")]

            config_dict[key] = value

    return config_dict               

def parse_model_config(config_dict):
    """Parse model configuration and store it in a ModelConfigC object."""
    model_config = ModelConfigC()
    model_config.model_path = config_dict["onnx"].encode("utf-8")
    model_config.num_class = int(config_dict["c"])
    model_config.score_threshold = float(config_dict["thresh"])
    
    # Convert anchors to a ctypes array\
    '''
    if not isinstance(config_dict["anchors"], list):
        raise ValueError("anchors should be a list of integers")
    '''
    if "anchors" in config_dict :
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
    model_config.detection_colormap = ctypes.cast(c_detection_colormap, ctypes.POINTER(ctypes.c_int))
    model_config.detection_colormap_size = len(detection_colormap)

    return model_config


def parse_inference_config(config_dict):
    """Parse inference configuration and store it in an InferenceConfigC object."""
    inference_config = InferenceConfigC()
    inference_config.precision = config_dict["precision"].encode("utf-8")
    inference_config.profile = False
    if "sparse" in config_dict :
        inference_config.sparse = bool(config_dict["sparse"])
    inference_config.dla_core_id = -1
    inference_config.use_first_layer = False
    inference_config.use_last_layer = False
    inference_config.batch_size = 1
    inference_config.scale = 1.0
    inference_config.calibration_images = config_dict["calibration_images"].encode("utf-8")
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
    if "sparse" in config_dict :    
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
    if "subnet_onnx" in config_dict :
        #Parse subnet like TLR and anonymizer
        subnet_model_config = parse_subnet_model_config(config_dict)
        subnet_inference_config = parse_inference_config(config_dict)
        if "batch_size" in config_dict :
            batch_size = config_dict["batch_size"]
            subnet_inference_config.max_batch_size = batch_size                
            subnet_inference_config.optimal_batch_size = (int)(batch_size/2)
            subnet_inference_config.min_batch_size = 1        
        return TrtLightnet(model_config, inference_config, build_config, subnet_model_config, subnet_inference_config)
    else :
        return TrtLightnet(model_config, inference_config, build_config)    

def parse_subnet_model_config(config_dict):
    """Parse model configuration and store it in a ModelConfigC object."""
    model_config = ModelConfigC()
    model_config.model_path = config_dict["subnet_onnx"].encode("utf-8")
    model_config.num_class = int(config_dict["subnet_c"])
    model_config.score_threshold = float(config_dict["subnet_thresh"])
    
    # Convert anchors to a ctypes array\
    '''
    if not isinstance(config_dict["anchors"], list):
        raise ValueError("anchors should be a list of integers")
    '''
    if "anchors" in config_dict :
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
    model_config.detection_colormap = ctypes.cast(c_detection_colormap, ctypes.POINTER(ctypes.c_int))
    model_config.detection_colormap_size = len(detection_colormap)

    return model_config
