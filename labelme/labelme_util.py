import sys
sys.path.append('../')

import os
import json

import common_util

def count_object_from_labelme(labelme_path):
    labelme = common_util.load_json(labelme_path)
    shapes = labelme.get("shapes", [])
    return len(shapes)


if __name__ == "__main__":

    # BASE_FOLDER = r"E:\DATA\@cctv\@labelme"
    # BASE_FOLDER = r"E:\DATA\@cctv\@labelme_검수\보행자_검수\보행자_검수완료"
    BASE_FOLDER = r"E:\DATA\@cctv\@labelme_검수\보행자_검수2차\2차 검수파일"
    
    all_count = 0
    for folder in os.listdir(BASE_FOLDER):
        folder_path = os.path.join(BASE_FOLDER, folder)
        for file in os.listdir(folder_path):
            if file.endswith(".json"):
                file_path = os.path.join(folder_path, file)
                count = count_object_from_labelme(file_path)
                print(file, count)
                all_count += count

    print("all_count:", all_count)

