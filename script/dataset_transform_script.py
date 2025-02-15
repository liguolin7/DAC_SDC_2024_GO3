import json
import os
from PIL import Image
#This script is used to convert the json tags of the training set into the txt format required by yolo
def convert_all_to_yolo_labels(json_folder, jpg_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for json_file in os.listdir(json_folder):
        if json_file.endswith('.json'):
            json_path = os.path.join(json_folder, json_file)
            base_filename = os.path.splitext(json_file)[0]
            jpg_file = base_filename + '.jpg'
            jpg_path = os.path.join(jpg_folder, jpg_file)
            if os.path.exists(jpg_path):
                with Image.open(jpg_path) as img:
                    img_width, img_height = img.size
                convert_to_yolo_label(json_path, img_width, img_height, os.path.join(output_folder, base_filename + '.txt'))

def convert_to_yolo_label(json_file, img_width, img_height, output_file):
    with open(json_file, 'r') as file:
        data = json.load(file)

    with open(output_file, 'w') as file:
        for item in data:

            if item['type'] <= 7:
                # Calculate the center x, y and the width and height in normalized coordinates
                x_center = (item['x'] + item['width'] / 2) / img_width
                y_center = (item['y'] + item['height'] / 2) / img_height
                width = item['width'] / img_width
                height = item['height'] / img_height

                # Write to file in YOLO format
                file.write(f"{item['type'] - 1} {x_center} {y_center} {width} {height}\n")
            elif 8 <= item['type'] <= 10:
                # Handle segmentation
                if item['segmentation']:
                    segmentation = item['segmentation'][0]
                    x_coords = segmentation[0::2]
                    y_coords = segmentation[1::2]
                    x_min = min(x_coords)
                    y_min = min(y_coords)
                    x_max = max(x_coords)
                    y_max = max(y_coords)

                    # Calculate the center x, y and the width and height in normalized coordinates
                    x_center = (x_min + x_max) / 2 / img_width
                    y_center = (y_min + y_max) / 2 / img_height
                    width = (x_max - x_min) / img_width
                    height = (y_max - y_min) / img_height

                    # Write to file in YOLO format
                    file.write(f"{item['type'] - 1} {x_center} {y_center} {width} {height}\n")



json_folder_path = ''
jpg_folder_path = ''
output_folder_path = ''
convert_all_to_yolo_labels(json_folder_path, jpg_folder_path, output_folder_path)