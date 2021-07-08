import sys
sys.path.append('../')
sys.path.append('../../')


import os
import json
import base64
import requests
import shutil

import cv2
import numpy as np

import common_util
import image_util


CLASS_NAMES = common_util.read_lines("../data/classes/coco.names")

print(CLASS_NAMES)

VALID_CLASSES = ["person"]
# VALID_CLASSES = ["person", "bicycle"]
# VALID_CLASSES = [0, 1]

# 바운딩 박스의 최소 높이 (이것보다 높이가 커야지만 labelme로 변환시킴)
MIN_HEIGHT = 50

def create_labelme_from_lines(image_path, labelme_path, label_lines):

    image = image_util.imread(image_path)
    width = image.shape[1]
    height = image.shape[0]

    labelme_dict = common_util.load_json("labelme_template.json")    

    shape_list = []
    for line in label_lines:
        # shape_dict = shape_dict_default.copy()
        # shape_dict["label"] = label_dict["class_text"]
        x1, y1, w, h, class_name = line.split(",")

        x1 = int(x1)
        y1 = int(y1)
        w = int(w)
        h = int(h)

        x2 = x1 + w
        y2 = y1 + h
        # x1 = x1 / W * width
        # y1 = y1 / H * height
        # x2 = x2 / W * width
        # y2 = y2 / H * height

        # class_id = label_dict["class_id"]
        # class_name = CLASS_NAMES[class_id]

        if class_name in VALID_CLASSES:
    
            shape_dict = {
                "label": class_name,
                "line_color": None,
                "fill_color": None,
                "points": [[x1, y1], [x2, y2]],
                "shape_type": "rectangle",
                "flags": {}
            }
            shape_list.append(shape_dict)

    if len(shape_list) == 0:
        # 유효한 객체가 없으면 처리 중단
        return

    labelme_dict["shapes"] = shape_list
    labelme_dict["imagePath"] = os.path.basename(image_path)
    labelme_dict["imageWidth"] = width
    labelme_dict["imageHeight"] = height

    # print(labelme_dict)
    common_util.save_json(labelme_path, labelme_dict)

    # output_folder = os.path.dirname(labelme_path)
    # shutil.copy(image_path, output_folder)


def create_labelme_from_rect(image_path, labelme_path, label_list):

    image = image_util.imread(image_path)
    width = image.shape[1]
    height = image.shape[0]

    labelme_dict = common_util.load_json("labelme_template.json")    

    shape_list = []
    # shape_dict_default = labelme_dict["shapes"][0]
    for label_dict in label_list:
        # shape_dict = shape_dict_default.copy()
        # shape_dict["label"] = label_dict["class_text"]
        box = label_dict["box"]
        x1 = box["x"]
        y1 = box["y"]
        x2 = box["x"] + box["w"]
        y2 = box["y"] + box["h"]
        # x1 = x1 / W * width
        # y1 = y1 / H * height
        # x2 = x2 / W * width
        # y2 = y2 / H * height

        class_id = label_dict["class_id"]
        class_name = CLASS_NAMES[class_id]

        if class_name in VALID_CLASSES:
    
            shape_dict = {
                # "label": "LP",
                "label": class_name,
                "line_color": None,
                "fill_color": None,
                "points": [[x1, y1], [x2, y2]],
                "shape_type": "rectangle",
                "flags": {}
            }
            shape_list.append(shape_dict)

    if len(shape_list) == 0:
        # 유효한 객체가 없으면 처리 중단

        return

    labelme_dict["shapes"] = shape_list
    labelme_dict["imagePath"] = os.path.basename(image_path)
    labelme_dict["imageWidth"] = width
    labelme_dict["imageHeight"] = height

    # print(labelme_dict)
    common_util.save_json(labelme_path, labelme_dict)

    output_folder = os.path.dirname(labelme_path)
    shutil.copy(image_path, output_folder)


if __name__ == "__main__":
    

    BASE_FOLDER = r"E:\DATA\@cctv\@labelme"
    OUTPUT_FOLDER = r"E:\DATA\@cctv\@labelme"

    # folders = ["20191109140542"]
    # folders = ["20191109151028"]
    # folders = ["20191218103400"]
    # folders = ["20200307191600"]
    # folders = ["20191109161514"]

    # folders = ["20191218225837", "20191109173924", "20191109194854", "20191218143806"]
    folders = ["20191218130000", "20191218150900", "20191218140416", "20191218115448", "20191218105004", "20191218111818"]
    
    for folder in folders:
        folder_path = os.path.join(BASE_FOLDER, folder)
        output_folder = os.path.join(OUTPUT_FOLDER, folder)
        common_util.check_folder(output_folder)

        for file in os.listdir(folder_path):
            print(file)
            image_path = os.path.join(folder_path, file)
            json_file = file + ".json"
            json_path = os.path.join(folder_path + "_json", json_file)
            label_lines = common_util.load_json(json_path)

            file_noext, _ = os.path.splitext(file)
            labelme_file = file_noext + ".json"
            labelme_path = os.path.join(output_folder, labelme_file)
            create_labelme_from_lines(image_path, labelme_path, label_lines)
