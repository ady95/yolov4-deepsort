import sys
import os

# subprocess 사용법: https://umbum.dev/382
import subprocess


OUTPUT_FOLDER = r"E:\DATA\@cctv\@labelme"
BASE_FOLDER = r"E:\DATA\@cctv\보행자"


video_files = os.listdir(BASE_FOLDER)
if len(video_files) == 0:
    print("비디오 파일이 존재하지 않습니다.")
    exit()


video_files.sort()
for video_file in video_files:
    video_path = os.path.join(BASE_FOLDER, video_file)
    vals = video_file.split("_")
    folder_name = vals[1]
    output_path = os.path.join(OUTPUT_FOLDER, folder_name)

    print("------------------------------------------------------")
    print("** 변환시작", video_path, "-->", output_path)
    print("------------------------------------------------------")

    commands = ["python", "object_tracker_labelme.py", "--video", video_path, "--output", output_path]

    print(" ".join(commands))

    subprocess.run(commands, shell=True)

    print("------------------------------------------------------")
    print("** 변환완료", output_path)
    print("------------------------------------------------------")

