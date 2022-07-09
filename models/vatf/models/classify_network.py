"""
Description       : 
Author            : NGUYEN QUANG HIEU
Maintainer        : NGUYEN QUANG HIEU
Date              : 04/03/2022 
Version           : 
Usage             :
Notes             :
"""

import tensorflow as tf
from resnet50_backbone import resnet50


class ClassifyNetWork(tf.keras.Model):
    def __init__(self):
        super(ClassifyNetWork, self).__init__()

        # Classify block
        self.classify_block = tf.keras.Sequential([
                                                    tf.keras.layers.Conv2D(128,kernel_size=(2, 2), activation='relu'),
                                                    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid'),
                                                    tf.keras.layers.Flatten(),
                                                    tf.keras.layers.Dense(64, activation="relu"),
                                                    tf.keras.layers.Dense(32, activation="relu"),
                                                    tf.keras.layers.Dense(2, activation="softmax")
                                                ])
    
    def call(self, image_array):
        result = self.classify_block(image_array) 
        return result
