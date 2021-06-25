import os
import cv2
import time
import pyautogui
import sys
import numpy as np

import pyparsing
import base64
import json
import requests
import tensorflow as tf

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
sys.path.append('C:\\build\OTJ2\\models\\research')
sys.path.append("C:\\build\OTJ2\\models\\research\\object_detection\\utils")
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

#ret, image = cap.read()

imageIn = "6623916aa8b4471651882f0d15a0ca2f.png"
URL = "http://localhost:8501/v1/models/ssdmobilenet:predict" 
IMAGE_URL = 'https://tensorflow.org/images/blogs/serving/cat.jpg'
PATH_TO_LABELS = 'C:\\build\\OTJ2\\Example1\\model\\mscoco_complete_label_map.pbtxt'


    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
#input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
#input_tensor = input_tensor[tf.newaxis,...]
#dl_request = requests.get(IMAGE_URL, stream=True)
#dl_request.raise_for_status()
#headers = {"content-type": "application/json"}
#image_content = cv2.imread(image,1).astype('uint8').tolist()
#body = {"instances":  input_tensor}
#print(body)
#r = requests.post(URL, data=body, headers = headers) 
#print('Prediction:', r['predictions'][0]['predicted_text'])
#print(r.text)
#This printed 'cat' on my consol

#jpeg_bytes = base64.b64encode(dl_request.content).decode('utf-8')
#predict_request = '{"instances" : [{"b64": "%s"}]}' % jpeg_bytes
#response = requests.post(URL, data=predict_request)
#print (response.text)
def proceedImage(image2):
    image_np = np.array(image2) 
    payload = {"instances": [image_np.tolist()]} 
    start = time.perf_counter() 
    res = requests.post(URL, json=payload) 
    end = time.perf_counter() 
    responseJson = json.loads(res.text)["predictions"][0]
    responseJson["detection_anchor_indices"] = np.array(responseJson["detection_anchor_indices"])
    responseJson["detection_multiclass_scores"] = np.array(responseJson["detection_multiclass_scores"])
    responseJson["detection_classes"] = np.array(responseJson["detection_classes"])
    responseJson["raw_detection_boxes"] = np.array(responseJson["raw_detection_boxes"])
    responseJson["detection_boxes"] = np.array(responseJson["detection_boxes"])
    responseJson["detection_scores"] = np.array(responseJson["detection_scores"])
    responseJson["raw_detection_scores"] = np.array(responseJson["raw_detection_scores"])
    num_detections = int(responseJson.pop('num_detections'))
    output_dict = responseJson
    #output_dict = {key: value[0, :num_detections]
    #               for key, value in responseJson.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    #Handle models with masks:
    if 'detection_masks' in output_dict:
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output_dict['detection_masks'], output_dict['detection_boxes'],
            image2.shape[0], image2.shape[1])      
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5, tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
    vis_util.visualize_boxes_and_labels_on_image_array(
        image2,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks_reframed', None),
        use_normalized_coordinates=True,
        line_thickness=8) 

while True:
    ret, image2 = cap.read()
    proceedImage(image2)
    cv2.imshow('cam picture', image2)    
    print("the end")
    if cv2.waitKey(5) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
#f = open("demofile2.txt", "a")
#f.write(res.text)
#f.close()
#print(res.content)
#change to your public IP print(f"Took {time.perf_counter()-start:.2f}s") pprint(res.json())