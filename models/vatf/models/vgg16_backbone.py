"""
Description       : 
Author            : NGUYEN QUANG HIEU
Maintainer        : NGUYEN QUANG HIEU
Date              : 04/03/2022 
Version           : 
Usage             :
Notes             :
"""

import cv2
import numpy as np
import tensorflow as tf

class VGG16Classify(tf.keras.models.Model):
    def __init__(self):
        super(VGG16Classify, self).__init__()
        self.backbone = tf.keras.applications.VGG16(weights="imagenet", include_top=False)
        self.flatten = tf.keras.layers.Flatten()
        self.dense64 = tf.keras.layers.Dense(64, activation="relu")
        self.dense32 = tf.keras.layers.Dense(32, activation="relu")
        self.dense7 = tf.keras.layers.Dense(7, activation="softmax")

    def call(self, image):
        # Classify model
        result = self.backbone(image)
        result = self.flatten(result)
        result = self.dense64(result)
        result = self.dense32(result)
        result = self.dense7(result)
        return result
        