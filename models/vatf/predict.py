"""
Description       : 
Author            : NGUYEN QUANG HIEU
Maintainer        : NGUYEN QUANG HIEU
Date              : 26/01/2021 
Version           : 
Usage             :
Notes             :
"""

import argparse

import numpy as np
import tensorflow as tf
import cv2

from utils.parser import get_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str,
                        help="Path to image need predict")
    parser.add_argument("--config_path", type=str, default="./configs/vatf.yaml",
                        help="Path to config file")
    parser.add_argument("--model_path", type=str, default="./output/best_model/",
                        help="Path to model")
    return parser.parse_args()


# if __name__ == "__main__":
#     args = parse_args()

#     # Load config
#     model = tf.keras.models.load_model(args.model_path)
    
#     # Load image
#     image = cv2.imread(args.image_path)
#     image = cv2.resize(image, dsize=(331,468), interpolation=cv2.INTER_AREA)
#     image = np.asarray(image)/255
#     image = np.array([image])

#     # Predict
#     predict_score = model(image)
#     pred_cls = tf.argmax(predict_score[0])
#     print("[Predict]: class -", np.array(pred_cls), "\tScore -", float(tf.reduce_max(predict_score)))
