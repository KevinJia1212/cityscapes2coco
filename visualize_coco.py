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

def showAnns(anns):
    if len(anns) == 0:
        return 0
    ax = plt.gca()
    ax.set_autoscale_on(False)
    captions = []
    polygons = []
    rectangles = []
    color = []
    for ann in anns:
        c = (np.random.random((1, 3))*0.6+0.4).tolist()[0]
        if 'segmentation' in ann:
            if type(ann['segmentation']) == list:
                # polygon
                for seg in ann['segmentation']:
                    captions.append(cat_names[ann['category_id']-1])
                    poly = np.array(seg).reshape((int(len(seg)/2), 2))
                    l_corner, w, h = (ann['bbox'][0], ann['bbox'][1]), ann['bbox'][2], ann['bbox'][3]
                    rectangles.append(Rectangle(l_corner, w, h))
                    polygons.append(Polygon(poly))
                    color.append(c)

    p = PatchCollection(rectangles, facecolor='none', edgecolors=color, alpha=1, linestyle='--', linewidths=2)
    ax.add_collection(p)

    for i in range(len(captions)):
        x = rectangles[i].xy[0]
        y = rectangles[i].xy[1]
        ax.text(x, y, captions[i], size=10, verticalalignment='top', color='w', backgroundcolor="none")

    p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.3)
    ax.add_collection(p)
    p = PatchCollection(polygons, facecolor='none', edgecolors=color, linewidths=2)
    ax.add_collection(p)

annfile = '/home/d205-kun/cityscapes/train/train.json'
imgroot = '/home/d205-kun/cityscapes/train/training_images'
coco = COCO(annfile)
cats = coco.loadCats(coco.getCatIds())
cat_names = [cat['name'] for cat in cats]
catids = coco.getCatIds(catNms='truck')
imgids = coco.getImgIds(catIds=catids)
img = coco.loadImgs(imgids[np.random.randint(0, len(imgids))])[0]
I = io.imread(os.path.join(imgroot, img['file_name']))
plt.imshow(I)
annids = coco.getAnnIds(imgIds=img['id'])
anns = coco.loadAnns(annids)
showAnns(anns)
plt.show()
    
