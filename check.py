import os
import sys
if "/opt/ros/kinetic/lib/python2.7/dist-packages" in sys.path:
    sys.path.remove("/opt/ros/kinetic/lib/python2.7/dist-packages")
import cv2
import numpy as np
from skimage import io
from matplotlib import pyplot as plt
from matplotlib import patches, lines
from matplotlib.patches import Polygon, Rectangle
from matplotlib.collections import PatchCollection
import json
from pycocotools.coco import COCO

class CityScapes:
    def __init__(self, trainimages, trainann, valimages, valann):
        self.train_image_path = trainimages
        train_ann = trainann
        self.val_image_path = valimages
        val_ann = valann
        # self.train_data = COCO(train_ann)
        self.val_data = COCO(val_ann)
        
    def ann_check(self, dataset):
        output_list = []
        imgids = dataset.getImgIds()
        for imgid in imgids:
            img = dataset.loadImgs(imgid)[0]
            annids = dataset.getAnnIds(imgIds=img['id'])
            # anns = coco.loadAnns(annids)
            if len(annids) == 0:
                print(img['file_name'])
                output_list.append(img['file_name'])
        return output_list
    
    def class_count(self, dataset):
        counts = {"car": 0, "pedestrian": 0, "truck": 0, "bus": 0, "rider": 0 }
        imgids = dataset.getImgIds()
        for imgid in imgids:
            img = dataset.loadImgs(imgid)[0]
            annids = dataset.getAnnIds(imgIds=img['id'])
            
            if len(annids) == 0:
                continue
            else:
                for id in annids:
                    ann = dataset.loadAnns(id)[0]
                    if ann['category_id'] == 1:
                        counts['car'] += 1
                    elif ann['category_id'] == 2:
                        counts['pedestrian'] += 1
                    elif ann['category_id'] == 3:
                        counts['truck'] += 1
                    elif ann['category_id'] == 4:
                        counts['bus'] += 1
                    elif ann['category_id'] == 5:
                        counts['rider'] += 1
        print(counts)

if __name__ == "__main__":
    cityscapes_root = '/home/kun/cityscapes_val'
    train_images = os.path.join(cityscapes_root, "train/training_images")
    train_anns = os.path.join(cityscapes_root, "train/train.json") 
    val_images = os.path.join(cityscapes_root, "images")
    val_anns = os.path.join(cityscapes_root, "val.json")

    cityscapes = CityScapes(train_images, train_anns, val_images, val_anns)
    cityscapes.class_count(cityscapes.val_data)
    # no_anns = cityscapes.ann_check(cityscapes.val_data)
    # for img in no_anns:
    #     os.remove(os.path.join(val_images, img))