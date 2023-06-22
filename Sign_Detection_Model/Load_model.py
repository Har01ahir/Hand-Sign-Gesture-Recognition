import os
from object_detection.utils import config_util
# from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
import tensorflow as tf

from Setup_Path import CHECKPOINT_PATH

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file("C:/Users/hardi/MyProjects/Project-Sign_Detection/Tensorflow/workspace/models/my_ssd_mobnet/pipeline.config")
detection_model = model_builder.build(model_config=configs['model'], is_training=True)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join("C:/Users/hardi/MyProjects/Project-Sign_Detection/Tensorflow/workspace/models/my_ssd_mobnet/", 'ckpt-6.index')).expect_partial()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections