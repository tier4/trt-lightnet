9# Copyright 2025 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import cv2
import pylightnet
from pathlib import Path
from typing import Dict, Any, Tuple

def parse_args():
    """Parse command-line arguments for video path and config file."""
    parser = argparse.ArgumentParser(
        description="Run inference on a video using PyLightNet."
    )
    parser.add_argument(
        "-t4d", "--t4dataset", required=True, type=str, help="Path to the input video file"
    )
    parser.add_argument(
        "-cam", "--camera-name", required=True, type=str, help="camera name"
    )    
    parser.add_argument(
        "-f",
        "--flagfile",
        required=True,
        type=str,
        help="Path to the configuration (flag) file",
    )
    return parser.parse_args()


def demo(t4d_path, config_path, CAM_NAME):
    """Run inference on the given video using the provided PyLightNet config.
    Args:
        video_path (str): Path to the input video.
        config_path (str): Path to the PyLightNet configuration file.
    """    

    config = pylightnet.load_config(config_path)
    names = pylightnet.load_names_from_file(config["names"])
    colormap = pylightnet.load_colormap_from_file(config["rgb"])
    mask_dict = pylightnet.load_segmentation_data(config["mask"])
    lightnet = pylightnet.create_lightnet_from_config(config)
    seg_data = (
        pylightnet.load_segmentation_data(config["mask"]) if "mask" in config else None
    )
    t4d_dir = Path(t4d_path)
    ann_dir = t4d_dir / "annotation"
    t4_data = t4d_dir / "data"
    print(ann_dir)
    
    calibrated_info: Dict[str, Any]
    calibrated_info, c_info = lightnet.get_calibrated_info(ann_dir, CAM_NAME)
    
    # 2. Iterate through each sensor directory inside 't4/data'
    try:
        lidar_data = t4_data / "LIDAR_CONCAT"
        # C++ Logic: 
        # 4. Iterate through each file within the selected sensor directory
        all_entries = list(lidar_data.iterdir())
        sorted_files = sorted([p for p in all_entries if p.is_file()], key=lambda p: p.name)
        for file_path in sorted_files:
            image = lightnet.make_range_image(file_path, c_info, 120.0)                        
            print(f"Inference from {file_path}")
                                        
            lightnet.infer(image, cuda=True)
            if seg_data is not None:
                argmax2bgr_ptr = lightnet.segmentation_to_argmax2bgr(seg_data)
                lightnet.make_mask(argmax2bgr_ptr)
                masks = lightnet.get_masks_from_cpp()
                for i, mask in enumerate(masks):
                    cv2.imshow(f"Mask_{i}", mask)

            # Entropy visualization
            if "entropy" in config:
                lightnet.make_entropy()
                entropies = lightnet.get_entropies()
                results = {}
                for index in sorted(mask_dict.keys()):
                    mask_name = mask_dict[index]["name"]
                    entropy = entropies[0][index]
                    results[mask_name] = entropy
                print("Entropy values:", results)

                entropy_maps = lightnet.get_entropy_maps_from_cpp()
                for i, emap in enumerate(entropy_maps):
                    cv2.imshow(f"Entropy_{i}", emap)

            image = cv2.resize(image, (1920, 1280))                    
            cv2.imshow("Range Image", image)                    
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break            
    except FileNotFoundError:
        print(f"Error: Base data directory not found: {t4_data}")    
    finally:
        lightnet.lib.free_calibrated_info(c_info)
        cv2.destroyAllWindows()
        
    lightnet.destroy()

    
if __name__ == "__main__":
    args = parse_args()
    demo(args.t4dataset, args.flagfile, args.camera_name)
