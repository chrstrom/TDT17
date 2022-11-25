import torch

import glob

import os


model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5n - yolov5x6, custom

path = './Norway/test/images/*.jpg'

predictions = []
for filename in glob.glob(path): 

    results = model(filename)
    image_id = os.path.basename(filename)
    bbox = results.pandas().xyxy[0].to_numpy()

    prediction_string = ""
    for object in bbox:
        object_class = object[5] + 1 # RDD assumes 1-indexed classes
        xmin = object[0]
        xmax = object[2]
        ymin = object[1]
        ymax = object[3]

        prediction_string += f"{object_class} {xmin} {ymin} {xmax} {ymax} "

    submission_string = image_id + ", " + prediction_string 

    predictions.append(submission_string)



with open("rdd_submissions.txt", "w") as file:
    for prediction in predictions:
        file.writelines(prediction + "\n")
