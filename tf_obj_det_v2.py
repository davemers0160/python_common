import numpy as np
import os
#import six.moves.urllib as urllib
import sys
import tensorflow as tf
import cv2

from collections import defaultdict
from io import StringIO
 
sys.path.append("..")

# these are from the tensorflow/models/research/object_detection/
from object_detection.utils import ops as utils_ops
from utils import label_map_util
from utils import visualization_utils as vis_util


MODEL_PATH = '/home/ros/'

# MSCOCO
MODEL_NAME = 'ssd_resnet50_v1_fpn'
PATH_TO_LABELS = os.path.join('/home/ros/models/research/object_detection/data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90

#Open Image
#MODEL_NAME = 'ssd_mobilenet_v2_oid'
#PATH_TO_LABELS = os.path.join('/home/ros/models/research/object_detection/data', 'oid_v4_label_map.pbtxt')
#NUM_CLASSES = 600

PATH_TO_CKPT = MODEL_PATH + MODEL_NAME + '/frozen_inference_graph.pb'

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


## load in the detection graph from the frozen checkpoint file
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

	
## Run the inference/detection and return the detection results
def run_inference_for_single_image(image_np, graph):
  with graph.as_default():
    with tf.Session(graph=graph) as sess:

	  # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
	  image_np_expanded = np.expand_dims(image_np, axis=0)
	  image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
	  
	  # Each box represents a part of the image where a particular object was detected.
	  boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
	  
	  # Each score represent how level of confidence for each of the objects.
	  # Score is shown on the result image, together with the class label.
	  scores = detection_graph.get_tensor_by_name('detection_scores:0')
	  classes = detection_graph.get_tensor_by_name('detection_classes:0')
	  num_detections = detection_graph.get_tensor_by_name('num_detections:0')
	  
	  # Actual detection.
	  (boxes, scores, classes, num_detections) = sess.run(
	    [boxes, scores, classes, num_detections],
	    feed_dict={image_tensor: image_np_expanded})
  
  return (boxes, scores, classes, num_detections)
  

## this is the main routine
def run_detection():

  image_path = "/home/ros/tf_obj_det/test.jpg"
  image = cv2.imread(image_path)
  
  #image = cv2.resize(image, (800,600))
  
  boxes, scores, classes, num_detections = run_inference_for_single_image(image, detection_graph)

  # Visualization of the results of a detection.
  vis_util.visualize_boxes_and_labels_on_image_array(
      image,
      np.squeeze(boxes),
      np.squeeze(classes).astype(np.int32),
      np.squeeze(scores),
      category_index,
      use_normalized_coordinates=True,
      line_thickness=8)
 

  cv2.imshow('object detection', cv2.resize(image, (800,600)))
  cv2.waitKey(-1)


run_detection()
  
  
