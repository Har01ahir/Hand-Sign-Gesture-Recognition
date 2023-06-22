import cv2
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util, visualization_utils as viz_utils
from  Load_model import detect_fn

# Load the object detection model and label map
label_map_path = "C:/Users/hardi/MyProjects/Project-Sign_Detection/Tensorflow/workspace/annotations/label_map.pbtxt"
category_index = label_map_util.create_category_index_from_labelmap(label_map_path, use_display_name=True)

# Read the image
image_path = "C:/Users/hardi/MyProjects/Project-Sign_Detection/Tensorflow/workspace/images/data/test/Hello-12.jpg"
image_np = cv2.imread(image_path)       

# Convert image to tensor
input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)

# Perform object detection
detections = detect_fn(input_tensor)

num_detections = int(detections.pop('num_detections'))
detections = {key: value[0, :num_detections].numpy()
              for key, value in detections.items()}
detections['num_detections'] = num_detections

# detection_classes should be ints.
detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

label_id_offset = 1
image_np_with_detections = image_np.copy()

viz_utils.visualize_boxes_and_labels_on_image_array(
    image_np_with_detections,
    detections['detection_boxes'],
    detections['detection_classes'] + label_id_offset,
    detections['detection_scores'],
    category_index,
    use_normalized_coordinates=True,
    max_boxes_to_draw=5,
    min_score_thresh=.5,
    agnostic_mode=False
)

cv2.imshow('object detection', cv2.resize(image_np_with_detections, (800, 600)))
cv2.waitKey(0)
cv2.destroyAllWindows()
# D:/Hardik/All/Dt.26.11.2022/mrg/01/IMG_9161.JPG