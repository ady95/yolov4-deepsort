import sys
import os

# subprocess 사용법: https://umbum.dev/382
import subprocess


# BASE_FOLDER = r"E:\DATA\@cctv\@자살시도"
BASE_FOLDER = r"E:\DATA\@cctv\@자살의심"
OUTPUT_FOLDER = BASE_FOLDER


folders = os.listdir(BASE_FOLDER)
if len(folders) == 0:
    print("비디오 파일이 존재하지 않습니다.")
    exit()


def get_child_files(folder_path):
    return os.listdir(folder_path)


# folders.sort()
for folder in folders:
    folder_path = os.path.join(BASE_FOLDER, folder)

    for video_file in os.listdir(folder_path):
        video_path = os.path.join(folder_path, video_file)
        file_noext, _ = os.path.splitext(video_file)
        txt_file = file_noext + ".txt"
        output_path = os.path.join(OUTPUT_FOLDER, folder, txt_file)
        if os.path.exists(output_path):
            print("*** 이미 생성 완료:", output_path)
            continue

        print("------------------------------------------------------")
        print("** 생성시작", video_path, "-->", output_path)
        print("------------------------------------------------------")

        # python object_tracker_darklabel.py --video "E:\DATA\@cctv\@자살자\20200201_마포상류_자살시도\마포상류05-1_20200201221000_20200201221400.avi" --output "E:\DATA\@cctv\@자살자\20200201_마포상류_자살시도\마포상류05-1_20200201221000_20200201221400.txt"
        commands = ["python", "object_tracker_darklabel.py", "--video", video_path, "--output", output_path]

        print(" ".join(commands))

        subprocess.run(commands, shell=True)

        print("------------------------------------------------------")
        print("** 변환완료", output_path)
        print("------------------------------------------------------")

