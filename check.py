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

from pycocotools.coco import COCO

class CityScapes:
    def __init__(self, trainimages, trainann, valimages, valann):
        self.train_image_path = trainimages
        train_ann = trainann
        self.val_image_path = valimages
        val_ann = valann
        self.train_data = COCO(train_ann)
        self.val_data = COCO(val_ann)
        
    def ann_check(self):
        imgids = self.dataset.getImgIds()
        for imgid in imgids:
            img = self.dataset.loadImgs(imgid)[0]
            annids = self.dataset.getAnnIds(imgIds=img['id'])
            # anns = coco.loadAnns(annids)
            if len(annids) == 0:
                print(img['file_name'])

if __name__ == "__main__":
    cityscapes_root = '/home/d205-kun/cityscapes/'
    train_images = os.path.join(cityscapes_root, "train/training_images")
    train_anns = os.path.join(cityscapes_root, "train/train.json'") 
    val_images = os.path.join(cityscapes_root, "val/val_images")
    val_anns = os.path.join(cityscapes_root, "val/val.json'")
    cityscapes = CityScapes(imgroot, annfile)
    cityscapes.ann_check()
# cats = coco.loadCats(coco.getCatIds())
# cat_names = [cat['name'] for cat in cats]
# catids = coco.getCatIds()
# imgids = coco.getImgIds()

# showAnns(anns)
# plt.show()