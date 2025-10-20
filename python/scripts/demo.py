# Copyright 2025 TIER IV, Inc.
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


def parse_args():
    """Parse command-line arguments for video path and config file."""
    parser = argparse.ArgumentParser(
        description="Run inference on a video using PyLightNet."
    )
    parser.add_argument(
        "-v", "--video", required=True, type=str, help="Path to the input video file"
    )
    parser.add_argument(
        "-f",
        "--flagfile",
        required=True,
        type=str,
        help="Path to the configuration (flag) file",
    )
    return parser.parse_args()


def demo(video_path, config_path):
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

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video at '{video_path}'")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape

        # Inference
        lightnet.infer(frame, cuda=True)

        # Draw bounding boxes
        bboxes = lightnet.get_bboxes()
        pylightnet.draw_bboxes_on_image(frame, bboxes, colormap, names)

        # Segmentation visualization
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
                mask_name = mask_dict[index]['name']
                entropy = entropies[0][index]
                results[mask_name] = entropy
            print("Entropy values:", results)
                
            entropy_maps = lightnet.get_entropy_maps_from_cpp()
            for i, emap in enumerate(entropy_maps):
                cv2.imshow(f"Entropy_{i}", emap)

        # Resize main image for display
        frame_resized = cv2.resize(frame, (1920, 1280))
        cv2.imshow("Result", frame_resized)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    lightnet.destroy()


if __name__ == "__main__":
    args = parse_args()
    demo(args.video, args.flagfile)
