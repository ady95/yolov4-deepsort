import os
import json
import requests
# from PIL import Image

import image_util
# from pprint import pprint

class ActionClassHelper:


    SERVER_URL = "http://192.168.10.10:9001/predict"  # action

    IMG_WIDTH = 224
    IMG_HEIGHT = 224

    def __init__(self, fps = 30):
        self.fps = fps

    
    def put(self, obj_id, action_box, frame)

        crop_box = {
            "x": action_box[0],
            "y": action_box[1],
            "w": action_box[2] - action_box[0],
            "h": action_box[3] - action_box[1],
        }
        img_crop = image_util.crop_image(frame, crop_box)
        img_crop = image_util.resize(img_crop)


    def predict_file(self, image_path):
        filename = os.path.basename(image_path)
        extra_dict = { "filename": filename}
        image = Image.open(image_path)
        return self.predict_image(image, extra_dict)
    
    def predict_image(self, image, extra_dict = None):
        image = image_util.to_image(image)
        b64_image = image_util.encode_base64(image)
        return self.predict_base64(b64_image, extra_dict)


    def predict_base64(self, b64_image, extra_dict = None):
        param_dict = {
            "base64_image": b64_image,
        }
        if extra_dict:
            param_dict.update(extra_dict)

        ret = requests.post(self.SERVER_URL, json=param_dict)
        print(ret.text)
        result = ret.json()
        return result


if __name__ == "__main__":
    # client = EFNetClient(api_type="direction")
    client = EFNetClient()

    # base_path = r"D:\DATA\@car\car_photo\carphoto_20190614"
    # base_path = r"D:\DATA\@car\car_fake\car_fake_test_20190701"
    # base_path = r"D:\DATA\@car\car_brand\front_crop"
    # base_path = r"D:\DATA\@croudworks\CCTV\@train\person_gender\male_riding"
    # find_path = base_path
    # base_path = r"D:\DATA\@car\@test_kolas\20200523\output\N_check_output\carplate"
    # files = ["26879792_002.jpg", "26879794_002.jpg", "26879794_003.jpg"]
    
    base_path = r"E:\DATA\@cctv\2dcnn_action\test\9.ClimbOver"
    files = os.listdir(base_path)

    for filename in files:
        filepath = os.path.join(base_path, filename)
        if os.path.splitext(filepath)[1] != ".jpg":
            continue

        b64_image = image_util.encode_base64_from_file(filepath)
        result = client.predict_base64(b64_image)

        
        print(filename, result)
        
        # if result["direction"] != 2:
        #     print(filename, result)
    #       if result["fake_id"] > 0:
    #           fake_results.append((filename, result))

    # pprint(fake_results)

