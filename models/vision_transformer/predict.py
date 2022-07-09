
import os
import time

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from utils.parser import get_config
from models.vision_transformer import vision_transformer_model
from utils.dataloader import DataLoader

config = get_config("./configs/config.yaml")
config_train = config.TRAIN
config_dataset = config.DATASET
model = vision_transformer_model(
                        input_shape=config_train.INPUT_SHAPE, 
                        image_size=config_train.IMAGE_SIZE, 
                        patch_size=config_train.PATCH_SIZE, 
                        transformer_layers=config_train.TRANSFORMER_LAYERS, 
                        num_heads=config_train.NUM_HEADS,
                        projection_dim=config_train.PROJECTION_DIM,
                        mlp_head_units=config_train.MLP_HEAD_UNITS,
                        num_classes=config_dataset.NUM_CLASSES
                    )
optimizer = tfa.optimizers.AdamW(learning_rate=config_train.LEARNING_RATE, weight_decay=config_train.WEIGHT_DECAY)
loss_method = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric1 = tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")
metric2 = tf.keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy")
model.compile(optimizer=optimizer, loss=loss_method, metrics=[metric1, metric2])

model.load_weights("./output/ViT_checkpoint.ckpt")
model.save("./output/vit")
labels = open(config_dataset.LABELS).read().split('\n')

image = cv2.imread("./data/stest/04baocaokiemtoan/04_2.jpg")
image = cv2.resize(image, (32,32), interpolation = cv2.INTER_AREA)
image = np.array([image])
start = time.time()
result = model.predict(image)
result = tf.nn.softmax(result)
print(time.time() - start)
print("Predict:", labels[np.argmax(result)])