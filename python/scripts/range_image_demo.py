# Copyright 2025 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import cv2
import pylightnet
import numpy as np
from pathlib import Path
from typing import Dict, Any


def parse_args():
    """Parse command-line arguments for video path, config file, and save options."""
    parser = argparse.ArgumentParser(
        description="Run inference on T4 Dataset using PyLightNet."
    )
    parser.add_argument(
        "-t4d",
        "--t4dataset",
        required=True,
        type=str,
        help="Path to the T4 dataset directory",
    )
    parser.add_argument(
        "-cam",
        "--camera-name",
        required=True,
        type=str,
        help="Camera name for calibration info",
    )
    parser.add_argument(
        "-f",
        "--flagfile",
        required=True,
        type=str,
        help="Path to the configuration (flag) file",
    )
    # New saving options
    parser.add_argument(
        "--save-range-image", action="store_true", help="Save the RangeImage."
    )
    parser.add_argument(
        "--save-segmentation",
        action="store_true",
        help="Save the RangeImageSegmentation results.",
    )
    parser.add_argument(
        "--save-uncertainty",
        action="store_true",
        help="Save the Uncertainty (Entropy) maps.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="output",
        help="Directory to save the results.",
    )
    return parser.parse_args()


def demo(args):
    """Run inference on the given T4 dataset using the provided PyLightNet config.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    t4d_path = args.t4dataset
    config_path = args.flagfile
    CAM_NAME = args.camera_name

    # Setup output directory if saving is enabled
    output_dir = Path(args.output_dir)
    if args.save_range_image or args.save_segmentation or args.save_uncertainty:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving results to: {output_dir.resolve()}")

    config = pylightnet.load_config(config_path)
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
    # Get calibration information
    calibrated_info, c_info = lightnet.get_calibrated_info(ann_dir, CAM_NAME)

    try:
        lidar_data = t4_data / "LIDAR_CONCAT"
        # Iterate through each file within the selected sensor directory (LIDAR_CONCAT)
        all_entries = list(lidar_data.iterdir())
        sorted_files = sorted(
            [p for p in all_entries if p.is_file()], key=lambda p: p.name
        )

        for file_path in sorted_files:
            file_stem = (
                file_path.stem
            )  # Get file name without extension, e.g., "1640995200000000000"

            # 1. Make Range Image
            image = lightnet.make_range_image(file_path, c_info, 120.0)
            print(f"Inference from {file_path}")

            # 2. Run inference
            lightnet.infer(image, cuda=True)

            # --- Segmentation Processing ---
            if seg_data is not None:
                argmax2bgr_ptr = lightnet.segmentation_to_argmax2bgr(seg_data)
                lightnet.make_mask(argmax2bgr_ptr)
                masks = lightnet.get_masks_from_cpp()

                # Display segmentation masks
                for i, mask in enumerate(masks):
                    cv2.imshow(f"Mask_{i}", mask)
                    # Save RangeImageSegmentation
                    if args.save_segmentation:
                        save_path = output_dir / f"{file_stem}_seg_{i}.png"
                        cv2.imwrite(str(save_path), mask)

            # --- Uncertainty (Entropy) Processing ---
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

                # Display entropy maps
                for i, emap in enumerate(entropy_maps):
                    cv2.imshow(f"Entropy_{i}", emap)
                    # Save Uncertainty (Entropy) map
                    if args.save_uncertainty:
                        # Convert to 8-bit image for saving (optional, as emap is usually float/double)
                        # We normalize the map to [0, 255] for better visualization/saving as PNG
                        emap_uint8 = cv2.normalize(
                            emap, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U
                        )
                        save_path = output_dir / f"{file_stem}_uncertainty_{i}.png"
                        cv2.imwrite(str(save_path), emap_uint8)

            # --- Range Image Processing ---

            # Save RangeImage
            if args.save_range_image:
                # Save the RangeImage. Since 'image' is typically a float range image,
                # we convert it to a 16-bit PNG (or similar format) to preserve data.
                # Max range is 120.0 (used in make_range_image). Scaling to 65535 for 16-bit.
                # Assuming the range values are non-negative.
                max_range = 120.0
                image_normalized = (image / max_range) * 65535
                image_uint16 = image_normalized.astype(np.uint16)
                save_path = output_dir / f"{file_stem}_range_image.png"
                cv2.imwrite(str(save_path), image_uint16)

            # Display Range Image
            image_display = cv2.resize(image, (1920, 1280))
            cv2.imshow("Range Image", image_display)

            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except FileNotFoundError:
        print(f"Error: Base data directory not found: {t4_data}")
    finally:
        # Free calibrated info memory
        lightnet.lib.free_calibrated_info(c_info)
        cv2.destroyAllWindows()

    # Destroy the lightnet instance
    lightnet.destroy()


if __name__ == "__main__":
    args = parse_args()
    demo(args)
