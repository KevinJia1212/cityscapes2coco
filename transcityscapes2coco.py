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
 
ROOT_DIR = '/home/kun/cityscapes_val'
IMAGE_DIR = os.path.join(ROOT_DIR, "images")
ANNOTATION_DIR = os.path.join(ROOT_DIR, "gt")
ANNOTATION_SAVE_DIR = os.path.join(ROOT_DIR, "annotations")
INSTANCE_DIR = os.path.join(ROOT_DIR, "instances") 
IMAGE_SAVE_DIR = os.path.join(ROOT_DIR, "val_images")

INFO = {
    "description": "Cityscapes_Instance Dataset",
    "url": "https://github.com/waspinator/pycococreator",
    "version": "0.1.0",
    "year": "2020",
    "contributor": "Kevin_Jia",
    "date_created": "2020-1-23 19:19:19.123456"
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
pic_scale = 1.0
h_bias = 1.0

def image_trans():
    img_subfolders = os.listdir(IMAGE_DIR)
    image_count = 0
    for sub in img_subfolders:
        # sub_path = sub + '/' + sub
        image_sub_path = os.path.join(IMAGE_DIR, sub)
        for image in os.listdir(image_sub_path):
            img_path = os.path.join(image_sub_path, image)
            ann_name = image.split('_')[0] + '_' + image.split('_')[1] + '_' + image.split('_')[2] + '_gtFine_instanceIds.png'
            ann_sub_path = os.path.join(ANNOTATION_DIR, sub)
            ann_path = os.path.join(ann_sub_path, ann_name)
            if os.path.exists(ann_path): 
                pic = cv2.imread(img_path)
                h, w = pic.shape[:2]
                new_w = w * pic_scale
                new_h = new_w / 2
                top = int((h_bias*h-new_h)/2)
                bottom = int((h_bias*h+new_h)/2)
                left = int((w-new_w)/2)
                right = int((w+new_w)/2)
                roi = pic[top:bottom, left:right]
                img_save_path = os.path.join(IMAGE_SAVE_DIR, image)
                cv2.imwrite(img_save_path, roi) 
                annotation = cv2.imread(ann_path, -1)
                ann_roi = annotation[top:bottom, left:right]
                ann_save_path = os.path.join(ANNOTATION_SAVE_DIR, ann_name)
                cv2.imwrite(ann_save_path, ann_roi)
            else:
                print(image + '  do not have instance annotation')
            print(image_count)
            image_count += 1

def data_loader():
    imgs = os.listdir(IMAGE_SAVE_DIR)
    masks_generator(imgs, ANNOTATION_SAVE_DIR)

def masks_generator(imges, ann_path):
    global idx
    pic_count = 0
    for pic_name in imges:
        image_name = pic_name.split('.')[0]
        ann_folder = os.path.join(INSTANCE_DIR, image_name)
        os.mkdir(ann_folder)
        annotation_name = pic_name.split('_')[0] + '_' + pic_name.split('_')[1] + '_' + pic_name.split('_')[2] + '_gtFine_instanceIds.png'
        # annotation_name = image_name + '_instanceIds.png'
        print(annotation_name)
        annotation = cv2.imread(os.path.join(ann_path, annotation_name), -1)
        h, w = annotation.shape[:2]
        ids = np.unique(annotation)
        for id in ids:
            if id in background_label:
                continue
            else:
                class_id = id // 1000
                if class_id == 26:
                    instance_class = 'car'
                elif class_id == 24:
                    instance_class = 'pedestrian' 
                elif class_id == 27:
                    instance_class = 'truck'
                elif class_id == 28:
                    instance_class = 'bus'
                elif class_id == 25:
                    instance_class = 'rider'
                else:
                    continue    
            instance_mask = np.zeros((h, w, 3),dtype=np.uint8)
            mask = annotation == id
            instance_mask[mask] = 255
            mask_name = image_name + '_' + instance_class + '_' + str(idx) + '.png'
            cv2.imwrite(os.path.join(ann_folder, mask_name), instance_mask)
            idx += 1
        pic_count += 1
        print(pic_count)
 
def json_generate():
    car = 0
    pedestrian = 0
    truck = 0
    bus = 0
    rider = 0
    files = os.listdir(IMAGE_SAVE_DIR)

    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    image_id = 1
    segmentation_id = 1

    # go through each image
    for image_filename in files:
        image_name = image_filename.split('.')[0]
        image_path = os.path.join(IMAGE_SAVE_DIR, image_filename)
        image = Image.open(image_path)
        image_info = pycococreatortools.create_image_info(
                image_id, os.path.basename(image_filename), image.size)
        coco_output["images"].append(image_info)
        print(image_filename)
        annotation_sub_path = os.path.join(INSTANCE_DIR, image_name)
        ann_files = os.listdir(annotation_sub_path)
        if len(ann_files) == 0:
            print("ao avaliable annotation")
            continue
        else:
            for annotation_filename in ann_files:
                annotation_path = os.path.join(annotation_sub_path, annotation_filename)
                for x in CATEGORIES:
                    if x['name'] in annotation_filename:
                        class_id = x['id']
                        break
                # class_id = [x['id'] for x in CATEGORIES if x['name'] in annotation_filename][0]
                if class_id == 1:
                    car += 1
                elif class_id == 2:
                    pedestrian += 1
                elif class_id == 3:
                    truck += 1
                elif class_id == 4:
                    bus += 1
                elif class_id == 5:
                    rider += 1
                else:
                    print('illegal class id')
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
            print(image_id)
 
    with open('{}/val_modified.json'.format(ROOT_DIR), 'w') as output_json_file:
        json.dump(coco_output, output_json_file)
    print(car, pedestrian, truck, bus, rider)
 
 
if __name__ == "__main__":
    # image_trans()
    # data_loader()
    # json_generate()
    
    
