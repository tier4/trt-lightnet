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
    """Parse command-line arguments for video path and config files."""
    parser = argparse.ArgumentParser(
        description="Run multi-stage inference on a video using PyLightNet."
    )
    parser.add_argument(
        "-v", "--video", help="Path to the input video file", required=True, type=str
    )
    parser.add_argument(
        "-f1",
        "--flagfile1",
        help="Path to the primary config file for PyLightNet",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-f2",
        "--flagfile2",
        help="Path to the secondary config file for PyLightNet",
        required=True,
        type=str,
    )
    return parser.parse_args()


def demo(video_path, config_path1, config_path2):
    """Perform multi-stage inference and optional blurring on the input video.

    Args:
        video_path (str): Path to the input video file.
        config_path1 (str): Path to the primary configuration file.
        config_path2 (str): Path to the secondary configuration file (e.g., for face detector).
    """
    # Load primary model and its configuration
    config_dict = pylightnet.load_config(config_path1)
    names = pylightnet.load_names_from_file(config_dict["names"])
    colormap = pylightnet.load_colormap_from_file(config_dict["rgb"])
    subnet_names = pylightnet.load_names_from_file(config_dict["subnet_names"])
    subnet_colormap = pylightnet.load_colormap_from_file(config_dict["subnet_rgb"])
    target = pylightnet.load_names_from_file(config_dict["target_names"])
    batch_size = config_dict["batch_size"]
    lightnet = pylightnet.create_lightnet_from_config(config_dict)

    # Load secondary model (e.g., face detector) and its configuration
    face_config_dict = pylightnet.load_config(config_path2)
    face_names = pylightnet.load_names_from_file(face_config_dict["names"])
    face_colormap = pylightnet.load_colormap_from_file(face_config_dict["rgb"])
    face_detector = pylightnet.create_lightnet_from_config(face_config_dict)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    while True:
        ret, image = cap.read()
        if not ret:
            break

        height, width, _ = image.shape

        # Perform inference
        lightnet.infer(image, cuda=True)
        face_detector.infer(image, cuda=True)

        # Get bounding boxes
        bboxes = lightnet.get_bboxes()
        face_bboxes = face_detector.get_bboxes()

        # Merge primary and face detector results
        bboxes = pylightnet.merge_bbox(bboxes, face_bboxes, names, face_names)

        # Perform subnet classification on filtered bboxes
        subnet_bboxes = lightnet.infer_subnet_batches_from_bboxes(
            bboxes, image, names, target, subnet_names, batch_size, min_crop_size=1
        )

        # Apply blur to sensitive regions
        image = pylightnet.blur_sensitive_regions(
            image, face_bboxes, face_names, (31, 31)
        )
        image = pylightnet.blur_sensitive_regions(
            image, subnet_bboxes, subnet_names, (31, 31)
        )

        # Draw results
        pylightnet.draw_bboxes_on_image(image, bboxes, colormap, names)
        pylightnet.draw_bboxes_on_image(
            image, subnet_bboxes, subnet_colormap, subnet_names
        )
        pylightnet.draw_bboxes_on_image(image, face_bboxes, face_colormap, face_names)

        # Display result
        image = cv2.resize(image, (1920, 1280))
        cv2.imshow("Result", image)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parse_args()
    demo(args.video, args.flagfile1, args.flagfile2)
