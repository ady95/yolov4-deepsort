import sys
sys.path.append('../')
sys.path.append('../../')

import os
import json
import base64
import requests

import cv2
import numpy as np

import common_util

class LabelmeHelper:

    def __init__(self):
        pass

    # ALPR 에서 나온 4개의 꼭지점으로부터 Labelme 파일을 생성함
    def create_file(self, points, image_filename, image_size, labelme_path):

        labelme_dict = common_util.load_json("labelme_template.json")

        labelme_dict["shapes"][0]["points"] = points

        labelme_dict["imagePath"] = image_filename
        labelme_dict["imageWidth"] = image_size[0]
        labelme_dict["imageHeight"] = image_size[1]

        print(labelme_dict)
        common_util.save_json(labelme_path, labelme_dict)


if __name__ == "__main__":
    labelmeHelper = LabelmeHelper()
    
    points = [[481, 292],
             [557, 263],
             [557, 285],
             [481, 314]]

    labelmeHelper.create_file(points, image_filename = "labelme_template_sample.jpg",
                            image_size = (611, 412), labelme_path = "labelme_template_sample.json")