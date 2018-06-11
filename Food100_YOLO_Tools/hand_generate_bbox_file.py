# food100_generate_bbox_file.py
#
# Read each food100 class image directory 'bb_info.txt' and create individual bbox files.
#
# This script creates 2 bounding box files for each food image:
# (1) The bounding box file that is saved along with the image in the same directory,
#     this allow to use bounding box editing tool. When the image re-edit,
#     it remembers the previous edited bounding boxes.
# (2) The bounding box file that is saved in 'labels' directory,
#     parallel to the image file and using darknet expected bbox format.
#     This allows darknet YOLO training with the expected bbox format.
#

import os
from PIL import Image

# import cv2  
import json
import numpy as np
import math
import pathlib


# modify the directories and class file to fit
datapath = 'images'
labelpath = 'labels'
classfilename = 'food100.names'

def convert_yolo_bbox(img_size, box):
    # img_bbox file is [0:img] [1:left X] [2:bottom Y] [3:right X] [4:top Y]
    dw = 1./img_size[0]
    dh = 1./img_size[1]
    x = (int(box[1]) + int(box[3]))/2.0
    y = (int(box[2]) + int(box[4]))/2.0
    w = abs(int(box[3]) - int(box[1]))
    h = abs(int(box[4]) - int(box[2]))
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    # Yolo bbox is center x, y and width, height
    return (x,y,w,h)

def generate_bbox_file(datapath, labelpath, classid, classname):
    dataDir = os.path.join(datapath, str(classid))
    labelDir = os.path.join(labelpath, str(classid))
    bb_filename = os.path.join(dataDir, 'bb_info.txt')
    if not os.path.exists(labelDir):
        os.makedirs(labelDir)
    with open(bb_filename) as fp:
        for line in fp.readlines():
            # img_bbox file is [0:img] [1:left X] [2:bottom Y] [3:right X] [4:top Y]
            img_bbox = line.strip('\n').split(' ')
            if img_bbox[0] != 'img':
                img_bbox_filename = os.path.join(dataDir, img_bbox[0]+'.txt')
                with open(img_bbox_filename, 'w') as f:
                    # [number of bbox]
                    # [left X] [top Y] [right X] [bottom Y] [class name]
                    f.write('1\n')
                    f.write('%s %s %s %s %s\n' %(img_bbox[1], img_bbox[4], img_bbox[3], img_bbox[2], classname))
                    f.close()

                image_filename = os.path.join(dataDir, img_bbox[0]+'.jpg')
                yolo_label_filename = os.path.join(labelDir, img_bbox[0]+'.txt')
                with open(yolo_label_filename, 'w') as f:
                    img = Image.open(image_filename)
                    yolo_bbox = convert_yolo_bbox(img.size, img_bbox)
                    if (yolo_bbox[2] > 1) or (yolo_bbox[3] > 1):
                        print("image %s bbox is " %(image_filename) + ' '.join(map(str, yolo_bbox)))
                    f.write(str(classid-1) + ' ' + ' '.join(map(str, yolo_bbox)) + '\n')
                    img.close()
                    f.close()
        fp.close()

# classid = 0
# classid2name = {}
# if os.path.exists(classfilename):
#     with open(classfilename) as cf:
#         for line in cf.readlines():
#             classname = line.strip('\n')
#             classid = classid + 1
#             classid2name[classid] = classname

# for id in classid2name.keys():
#     print("generating %d %s" %(id, classid2name[id]))
#     generate_bbox_file(datapath, labelpath, id, classid2name[id])


# constants for calculations
imgW = 1920 #our input images have the same dimensions
imgH = 1080

def convert_yolo_annotations(indir: str, outdir: str, padding=0, limit=0) -> None:
    """
    Crops untouched images from the Assignment 3 dataset to show only hands.
    Location of only the hand is determined through the minimum and maximum x/y coordinates of the joint labellings.
    For each image, left (_L) and right (_R) images are saved to `outdir` (each untouched image has 2 hands in it) -
    i.e., 2 * `limit` images will be generated.
    Expects `indir` to be a directory containing:
      - annotation.json
      - Color/ (a subdirectory containing all the untouched images)

    `outdir` is the directory to save all cropped images.
    `padding` is the number of pixels appended to each images' border, in case annotation data is too narrow.
    `limit` is the number of untouched images to process. If <= 0, all images will be processed.
    """
    root = pathlib.Path(indir)
    assert root.is_dir(), 'expected directory "{0}" to exist, with "annotation.json" and "Color/" within.'.format(indir)

    annotations = {}
    with open('{0}/annotation.json'.format(root)) as infile:
        annotations = json.load(infile)

    extension = 'jpg'
    i = 0
    imgFiles = os.listdir(os.path.join(indir, "Color"))

    # for name, data in annotations.items():
    for name in imgFiles:
        fileName, ext = name.split(".")

        annotationsForFile = []
        annotations.get(fileName + "_L") and annotationsForFile.append(annotations.get(fileName + "_L"))
        annotations.get(fileName + "_R") and annotationsForFile.append(annotations.get(fileName + "_R"))

        write_path = pathlib.Path('{0}/Color/{1}.txt'.format(indir, fileName)) 
        yoloOutputFile = open(write_path, "w")

        for index, data in enumerate(annotationsForFile):
            if limit > 0 and i >= limit:
                return

            min_x, min_y = math.inf, math.inf
            max_x, max_y = 0, 0
            for x, y in data:
                min_x = min(x, min_x)
                min_y = min(y, min_y)
                max_x = max(x, max_x)
                max_y = max(y, max_y)

            min_x -= padding
            min_y -= padding
            max_x += padding
            max_y += padding

            """
                We need this format:
                <object-class-id> <x> <y> <width> <height>
                Where   
                    <x> and <y> are coords of center of box
                    <object-class-id> in an int
                    <x> <y> <width> <height> are floats from 0.0 to 1.0 inclusive

            """

            dW = 1.0 / imgW
            dH = 1.0 / imgH

            x = (int(min_x) + int(max_x)) / 2.0 * dW
            y = (int(min_y) + int(max_y)) / 2.0 * dH
            w = abs(int(max_x) - int(min_x)) * dW
            h = abs(int(max_y) - int(min_y)) * dH

            classID = 0 # we only have the hand class to look for

            print("{:}: {:} {:} {:} {:} {:}\n".format(fileName, classID, x, y, w, h))
            yoloOutputFile.write("{:} {:} {:} {:} {:}\n".format(classID, x, y, w, h))
            i += 1
        yoloOutputFile.close()

# Create all bounding box files in Yolo format: 
# convert_yolo_annotations(r"../Dataset", "", padding=0, limit=10)