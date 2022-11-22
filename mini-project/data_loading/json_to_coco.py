from tqdm import tqdm
import os
import natsort
import json

from collections import Counter
from sklearn.model_selection import train_test_split

from data_to_json import get_ann_file_names

from config import *

# Ensure that the directory from config actually exists, if not make it
if not os.path.exists(COCO_DIR): 
    os.makedirs(COCO_DIR)


def coco_category_list(category, supercat, id):
    dictionary={}
    dictionary["id"] = id
    dictionary["name"] = category
    dictionary["supercategory"] = supercat
    return dictionary

def to_coco_format(data):
    coco_format = {}
    coco_format['images'] = []
    coco_format['annotations'] = []

    id = 0

    for idx, file_name in tqdm(enumerate(data)):
        id += 1 

        with open(os.path.join(JSON_DIR,file_name), 'r') as f:
            json_data = json.load(f)

        image_dict = {}
        image_dict['id'] = idx
        image_dict['width'] = json_data["image"]["resolution"][0]
        image_dict['height'] = json_data["image"]["resolution"][1]
        image_dict['file_name'] = json_data["image"]["filename"]

        coco_format['images'].append(image_dict)

        for idy, instance in enumerate(json_data["annotations"]):
            id+=1
            if "box" in instance.keys():

                instance_dict = {} 
                instance_dict['id'] = id
                instance_dict['image_id'] = idx
                instance_dict['category_id'] = CLASSES.index(instance["class"])+1
                instance_dict['segmentation'] = [[]]
                
                instance_bbox = []
                instance_bbox.append(instance["box"][0])
                instance_bbox.append(instance["box"][1])
                instance_bbox.append(abs(instance["box"][2] - instance["box"][0]))
                instance_bbox.append(abs(instance["box"][3] - instance["box"][1]))

                instance_dict['bbox'] = instance_bbox
                instance_dict['area'] = instance_bbox[2]*instance_bbox[3]

                instance_dict['iscrowd'] = 0

                coco_format['annotations'].append(instance_dict)
        
    # categories
    categories = []
    for idx, label in enumerate(CLASSES):
        categories.append(coco_category_list(label, SUPERCAT[idx], idx+1))
    coco_format['categories'] = categories

    return coco_format

def save_coco_json(path, data):
    with open(path, 'w', encoding='utf-8') as make_file:
        json.dump(data, make_file, indent="\t")

category_counter_dict = {c: 0 for c in CLASSES}
category_filename_dict = {c: [] for c in CLASSES}

file_list = natsort.natsorted(get_ann_file_names(JSON_DIR))

roads_with_no_damage = []
for idx,file_name in tqdm(enumerate(file_list)):

    with open(os.path.join(JSON_DIR,file_name), 'r') as f:
        json_data = json.load(f)

    class_id_list = []

    for idy,instance in enumerate(json_data["annotations"]):
        class_id_list.append(instance["class"])

    count_items = Counter(class_id_list)
    
    try:
        class_id = count_items.most_common(n=1)[0][0]
        category_counter_dict[class_id] += 1
        category_filename_dict[class_id].append(file_name)
    except:
        roads_with_no_damage.append(file_name)

# Split train to train/validation with equal ratios for each class
train = [] 
validation = [] 

for key, value in category_counter_dict.items():

    if value == 0:
        continue

    files_containing_key = category_filename_dict[key]

    train_temp, validation_temp = train_test_split(files_containing_key, test_size=0.2)

    train.extend(train_temp)
    validation.extend(validation_temp)




save_coco_json(os.path.join(COCO_DIR, TRAINING_SET), to_coco_format(train))
save_coco_json(os.path.join(COCO_DIR, VALIDATION_SET), to_coco_format(validation))