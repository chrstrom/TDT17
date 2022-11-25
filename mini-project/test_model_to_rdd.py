import torch

import glob

import os
 
from tqdm import tqdm

model = torch.hub.load('ultralytics/yolov5', 'custom', path="best_model.pt",force_reload=True,autoshape=True)

path = './Norway/test/images/*.jpg'

predictions = []
for filename in tqdm(glob.glob(path)): 

    results = model(filename)
    image_id = os.path.basename(filename)
    bbox = results.pandas().xyxy[0].to_numpy()

    prediction_string = ""
    for object in bbox:
        object_class = int(object[5] + 1) # RDD assumes 1-indexed classes
        xmin = int(round(object[0], 0))
        xmax = int(round(object[2], 0))
        ymin = int(round(object[1], 0))
        ymax = int(round(object[3], 0))

        prediction_string += f"{object_class} {xmin} {ymin} {xmax} {ymax} "

    submission_string = image_id + ", " + prediction_string 

    predictions.append(submission_string)


with open("rdd_submissions.txt", "w") as file:
    for prediction in predictions:
        file.writelines(prediction + "\n")
