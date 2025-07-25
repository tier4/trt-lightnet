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
        description="Run multi-stage inference on a video using PyLightNet."
    )
    parser.add_argument(
        "-v", "--video", help="Path to the video file", required=True, type=str
    )
    parser.add_argument(
        "-f", "--flagfile", help="Path to the config file", required=True, type=str
    )
    return parser.parse_args()


def demo(video_path, config_path):
    """Perform multi-stage inference and optional blurring on the given video.

    Args:
        video_path (str): Path to the input video file.
        config_path (str): Path to the configuration file for PyLightNet.
    """
    config_dict = pylightnet.load_config(config_path)
    names = pylightnet.load_names_from_file(config_dict["names"])
    colormap = pylightnet.load_colormap_from_file(config_dict["rgb"])
    subnet_names = pylightnet.load_names_from_file(config_dict["subnet_names"])
    subnet_colormap = pylightnet.load_colormap_from_file(config_dict["subnet_rgb"])
    target = pylightnet.load_names_from_file(config_dict["target_names"])

    lightnet = pylightnet.create_lightnet_from_config(config_dict)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    while True:
        ret, image = cap.read()
        if not ret:
            break

        height, width, _ = image.shape

        # Perform multi-stage inference with target classes
        lightnet.infer_multi_stage(image, target, cuda=True)

        # Apply blur to regions detected by the multi-stage network
        lightnet.blur_image(image)

        # Draw bounding boxes from main and subnet detections for debug
        bboxes = lightnet.get_bboxes()
        pylightnet.draw_bboxes_on_image(image, bboxes, colormap, names)

        subnet_bboxes = lightnet.get_subnet_bboxes()
        pylightnet.draw_bboxes_on_image(
            image, subnet_bboxes, subnet_colormap, subnet_names
        )

        cv2.imshow("Result", image)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parse_args()
    demo(args.video, args.flagfile)
