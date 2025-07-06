import json
from pathlib import Path
from tqdm import tqdm
import shutil

json_path = "synthetic_data/labels_coco.json"
image_in_path = "synthetic_data/images/"
json_file = open(json_path)
json_data = json.loads(json_file.read())

image_folder = "labeled_synthetic_data/images/"
label_folder = "labeled_synthetic_data/labels/"

im_width = 1280
im_height = 720

Path(image_folder).mkdir(parents=True, exist_ok=True)
Path(label_folder).mkdir(parents=True, exist_ok=True)

for annotation in tqdm(json_data["annotations"]):
    image_id = annotation["image_id"]

    if json_data["images"][image_id]["id"] != image_id:
        print("image_id position assumption did not hold!")

    image_name = json_data["images"][image_id]["file_name"]
    label_name = image_name[:-4] + ".txt"

    yolo_x = (annotation["bbox"][0] + annotation["bbox"][2] / 2) / im_width
    yolo_y = (annotation["bbox"][1] + annotation["bbox"][3] / 2) / im_height
    yolo_w = annotation["bbox"][2] / im_width
    yolo_h = annotation["bbox"][3] / im_height

    yolo_line = f"{annotation['category_id']} {yolo_x} {yolo_y} {yolo_w} {yolo_h}"

    with open(label_folder + label_name, "a") as f:
        f.write(yolo_line + '\n')
    
    shutil.copy(image_in_path + image_name, image_folder)

