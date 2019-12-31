# cityscapes2coco
some Python scripts can be used to convert the cityscapes dataset to the annotation style of coco dataset for instance segmentation task.
transcityscapes2coco.py has some details that users should modify, like class definations and file paths of your cityscapes data.
visualize_coco.py can be used to visualize pictures and annotations of your converted dataset.

USAGE:
1.Before to run the scripts, you need to modify ROOT_DIR, IMAGE_DIR, ANNOTATION_DIR and create a INSTANCE_DIR under the ROOT_DIR to save binary masks of each instances.
2.According to the category defination of cityscapes dataset, we only keep 5 classes from all. If you want to define your own categories, you should modify the CATAGORY and corresponding scripts. That will be easy, because the scripts are simple.
3.Modify the file name of the .json output file at the end of transform script.
4.Run the scripts, and you can see some outputs in the terminal.
5.When finished, you can visualize the converted dataset by modify some path of files you just created    
