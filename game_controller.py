import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
import cv2
import time
import pyautogui
import pydirectinput
import tensorflow as tf
import sys
import numpy as np
from tensorflow.python.saved_model.load import load
sys.path.append("research")
sys.path.append("research\\object_detection\\utils")
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
####################################################################

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

PATH_TO_LABELS = 'labelmap.pbtxt'
PATH_TO_MODEL = 'savedModel'
#alternative network for detection with a wide range of objects
#PATH_TO_LABELS = 'ssdmobilenet\\mscoco_complete_label_map.pbtxt'
#PATH_TO_MODEL = 'ssdmobilenet'

###################################################################
#load model
def load_model(model_path):
    model = tf.saved_model.load(model_path)
    return model

###################################################################
#recognition for one image
def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis,...]
    
    # Run inference
    output_dict= model(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
   
    # Handle models with masks:
    if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                                    output_dict['detection_masks'], output_dict['detection_boxes'],
                                    image.shape[0], image.shape[1])      
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5, tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
    
    return output_dict


# List of the strings that is used to add correct label for each box.
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

#load model
loadedModell = load_model(PATH_TO_MODEL)

#####################################
# processing loop; runs endless until "q" is pushed
# classes:
# 1 == tennisball
# 2 == finger
# 3 == laugh

while True:
    ret, image = cap.read()
    output_dict = run_inference_for_single_image(loadedModell, image)

#######################START OF good place for your modifications #############################
    labelList = []
    index = 0

    for detection in output_dict['detection_scores']:
        if(detection > 0.50):
            labelNumber = output_dict['detection_classes'][index]
            labelList.append(labelNumber)
            print(labelNumber)
        else:
            break
        index+=1

    #tennisball
    if 1 in labelList:
        print("tennisball found; do tennisball things")

    #finger
    if 2 in labelList:
        print("finger found; do finger things")
        pydirectinput.keyDown("right")
    else:
        pydirectinput.keyUp("right")

    #laugh   
    if 3 in labelList:
        print("laugh found; do laugh things")
        pydirectinput.press("up")
    
####################### END OF good place for your modifications #############################

    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks_reframed', None),
        use_normalized_coordinates=True,
        line_thickness=8)
    cv2.imshow('cam picture', image) 
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

