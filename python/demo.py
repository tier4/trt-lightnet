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
import argparse
import cv2
import numpy as np
import ctypes
import time
from pathlib import Path
import os

import pylightnet

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v",
        "--video",
        help="video path",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-f",
        "--flagfile",
        help="config path",
        required=True,
        type=str,
    )    

    args = parser.parse_args()

    return args

def demo(video_path, config_path) :
    config_dict = pylightnet.load_config(config_path)        
    names = pylightnet.load_names_from_file(config_dict["names"])
    colormap = pylightnet.load_colormap_from_file(config_dict["rgb"])
    #load lightnet
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

        lightnet.infer(image, cuda=True)
            
        bboxes = lightnet.get_bboxes()
        pylightnet.draw_bboxes_on_image(image, bboxes, colormap, names)
        image = cv2.resize(image, (1920, 1280))

        cv2.imshow("Result", image)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()        
            
if __name__ == "__main__":
    args = parse_args()
    
    video_path = args.video
    config_path = args.flagfile    
    demo(video_path, config_path)


