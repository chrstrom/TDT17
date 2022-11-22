import xml.etree.ElementTree as ET
from tqdm import tqdm
import os
import natsort

import json

from config import *


if not os.path.exists(JSON_DIR): 
    os.makedirs(JSON_DIR)


def get_ann_file_names(ANN_DIR):
    
    file_format = ['xml','json']
    
    file_names = os.listdir(ANN_DIR)
    ann_file_names = []
    for file_name in tqdm(file_names):
        if file_name.split('.')[-1] in file_format:
            ann_file_names.append(file_name)
   
    return natsort.natsorted(ann_file_names)

ann_file_names = get_ann_file_names(ANN_DIR)

for ann_file in tqdm(ann_file_names):
    tree = ET.parse(os.path.join(ANN_DIR,ann_file))
    root = tree.getroot()
    
    if root.findtext('segmented') == '1':
        print('Warning')
        break
        
    filename = root.findtext('filename')
    img_w = root.find('size').findtext('width')
    img_h = root.find('size').findtext('height')
    objs = root.findall('object')
    
    gt = []
    for obj in objs:
        temp = {}

        cls_name = obj.findtext('name')
        temp['cls_name'] = cls_name
        
        bbox = obj.find('bndbox')
        x1 = int(float(bbox.findtext('xmin')))
        y1 = int(float(bbox.findtext('ymin')))
        x2 = int(float(bbox.findtext('xmax')))
        y2 = int(float(bbox.findtext('ymax')))
        temp['bbox'] = [x1,y1,x2,y2]
        
        if x2 <= x1 or y2 <= y1:
            print(f"Box size error !: {x1, y1, x2, y2}")
            break
        elif cls_name not in CLASSES:
            # print(f"Delete class: {cls_name}")
            break
        else:
            gt.append(temp)


ann_file_names = get_ann_file_names(ANN_DIR)

for idx,ann_file in enumerate(ann_file_names):
    
    tree = ET.parse(os.path.join(ANN_DIR,ann_file))
    root = tree.getroot()
    
    filename = root.findtext('filename')
    img_w = int(root.find('size').findtext('width'))
    img_h = int(root.find('size').findtext('height'))
    objs = root.findall('object')
    
    # Declare saved json form
    save_json = {}
    image_dict = {}
    image_dict['filename'] = filename
    image_dict['resolution'] = [img_w, img_h]
    
    
    annotation_dict = []
    for obj in objs:
        annotation_dict_temp = {}

        cls_name = obj.findtext('name')
        
        bbox = obj.find('bndbox')
        x1 = int(float(bbox.findtext('xmin')))
        y1 = int(float(bbox.findtext('ymin')))
        x2 = int(float(bbox.findtext('xmax')))
        y2 = int(float(bbox.findtext('ymax')))
        
        if x2 <= x1 or y2 <= y1:
            print(f"Box size error !: {x1, y1, x2, y2}")
            pass
        elif cls_name not in CLASSES:
            # print(f"Delete class: {cls_name}")
            pass
        else:
            annotation_dict_temp['box'] = [x1,y1,x2,y2]
            annotation_dict_temp['class'] = cls_name
            annotation_dict.append(annotation_dict_temp)
            
    save_json['image'] = image_dict
    save_json['annotations'] = annotation_dict
    
    save_path = os.path.join(JSON_DIR, filename.split('.')[0]+'.json')
    with open(save_path, 'w', encoding='utf-8') as make_file:
        json.dump(save_json, make_file, indent="\t")

