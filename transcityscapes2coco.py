import sys
if "/opt/ros/kinetic/lib/python2.7/dist-packages" in sys.path:
    sys.path.remove("/opt/ros/kinetic/lib/python2.7/dist-packages")

import cv2
import numpy as np
import os, glob
from shutil import copyfile
import datetime
import json
import os
import re
import fnmatch
from PIL import Image
import numpy as np
from pycococreatortools import pycococreatortools
 
ROOT_DIR = '/home/d205-kun/cityscapes/val'
IMAGE_DIR = os.path.join(ROOT_DIR, "images")
ANNOTATION_DIR = os.path.join(ROOT_DIR, "gt")
INSTANCE_DIR = os.path.join(ROOT_DIR, "instances") 
IMAGE_SAVE_DIR = os.path.join(ROOT_DIR, "val_images")

INFO = {
    "description": "Cityscapes_Instance Dataset",
    "url": "https://github.com/waspinator/pycococreator",
    "version": "0.1.0",
    "year": "2020",
    "contributor": "Kevin_Jia",
    "date_created": "2020-1-3 19:19:19.123456"
}
 
LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]

CATEGORIES = [
    {
        'id': 1,
        'name': 'car',
        'supercategory': 'cityscapes',
    },
        {
        'id': 2,
        'name': 'pedestrian',
        'supercategory': 'cityscapes',
    },
    {
        'id': 3,
        'name': 'truck',
        'supercategory': 'cityscapes',
    },
    {
        'id': 4,
        'name': 'bus',
        'supercategory': 'cityscapes',
    },
    {
        'id': 5,
        'name': 'rider',
        'supercategory': 'cityscapes',
    }
]

background_label = list(range(-1, 24, 1)) + list(range(29, 34, 1))
idx=0

def image_trans():
    img_subfolders = os.listdir(IMAGE_DIR)
    image_count = 0
    for sub in img_subfolders:
        sub_path = os.path.join(IMAGE_DIR, sub)
        for images in os.listdir(sub_path):
            img_path = os.path.join(sub_path, images)
            img_save_path = os.path.join(IMAGE_SAVE_DIR, images)
            copyfile(img_path, img_save_path)
            print(image_count)
            image_count += 1

def data_loader():
    img_subfolders = os.listdir(IMAGE_DIR)
    # image_count = 0
    for sub in img_subfolders:
        img_sub_path = os.path.join(IMAGE_DIR, sub)
        ann_sub_path = os.path.join(ANNOTATION_DIR, sub)
        masks_generator(os.listdir(img_sub_path), ann_sub_path)
        # for images in os.listdir(sub_path):
        #     img_path = os.path.join(sub_path, images)
        #     img_save_path = os.path.join(IMAGE_SAVE_DIR, images)
        #     copyfile(img_path, img_save_path)
        #     print(image_count)
        #     image_count += 1

def masks_generator(imges, ann_sub_path):
    global idx
    for pic_name in imges:
        annotation_name = pic_name.split('_')[0] + '_' + pic_name.split('_')[1] + '_' + pic_name.split('_')[2] + '_gtFine_instanceIds.png'
        print(annotation_name)
        annotation = cv2.imread(os.path.join(ann_sub_path, annotation_name), -1)
        name = pic_name.split('.')[0]
        h, w = annotation.shape[:2]
        ids = np.unique(annotation)
        for id in ids:
            if id in background_label:
                continue
            instance_id = id
            class_id = instance_id // 1000
            if class_id == 24:
                instance_class = 'pedestrian'
            elif class_id == 25:
                instance_class = 'rider' 
            elif class_id == 26:
                instance_class = 'car'
            elif class_id == 27:
                instance_class = 'truck'
            elif class_id == 28:
                instance_class = 'bus'
            else:
                continue
            print(instance_id)
            instance_mask = np.zeros((h, w, 3),dtype=np.uint8)
            mask = annotation == instance_id
            instance_mask[mask] = 255
            mask_name = name + '_' + instance_class + '_' + str(idx) + '.png'
            cv2.imwrite(os.path.join(INSTANCE_DIR, mask_name), instance_mask)
            idx += 1

def filter_for_pic(files):
    file_types = ['*.jpeg', '*.jpg', '*.png']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    files = [f for f in files if re.match(file_types, f)]
    # files = [os.path.join(root, f) for f in files]
    return files
 
def filter_for_instances(root, files, image_filename):
    file_types = ['*.png']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    files = [f for f in files if re.match(file_types, f)]
    basename_no_extension = os.path.splitext(os.path.basename(image_filename))[0]
    file_name_prefix = basename_no_extension + '.*'
    # files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_name_prefix, os.path.splitext(os.path.basename(f))[0])]
    return files
 
 
def json_generate():
    # for root, _, files in os.walk(ANNOTATION_DIR):
    files = os.listdir(IMAGE_SAVE_DIR)
    image_files = filter_for_pic(files)
    # masks_generator(image_files)
    # data_loader()
    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }
    image_id = 1
    segmentation_id = 1
    
    files = os.listdir(INSTANCE_DIR)
    instance_files = filter_for_pic(files)

    # go through each image
    for image_filename in image_files:
        image_path = os.path.join(IMAGE_SAVE_DIR, image_filename)
        image = Image.open(image_path)
        image_info = pycococreatortools.create_image_info(
                image_id, os.path.basename(image_filename), image.size)
        coco_output["images"].append(image_info)

        # filter for associated png annotations
        # for root, _, files in os.walk(INSTANCE_DIR):
        annotation_files = filter_for_instances(INSTANCE_DIR, instance_files, image_filename)

        # go through each associated annotation
        for annotation_filename in annotation_files:
            annotation_path = os.path.join(INSTANCE_DIR, annotation_filename)
            print(annotation_path)
            class_id = [x['id'] for x in CATEGORIES if x['name'] in annotation_filename][0]

            category_info = {'id': class_id, 'is_crowd': 'crowd' in image_filename}
            binary_mask = np.asarray(Image.open(annotation_path)
                                        .convert('1')).astype(np.uint8)

            annotation_info = pycococreatortools.create_annotation_info(
                    segmentation_id, image_id, category_info, binary_mask,
                    image.size, tolerance=2)

            if annotation_info is not None:
                coco_output["annotations"].append(annotation_info)

                segmentation_id = segmentation_id + 1

        image_id = image_id + 1
 
    with open('{}/val.json'.format(ROOT_DIR), 'w') as output_json_file:
        json.dump(coco_output, output_json_file)
 
 
if __name__ == "__main__":
    # image_trans()
    # data_loader()
    json_generate()
    
    
