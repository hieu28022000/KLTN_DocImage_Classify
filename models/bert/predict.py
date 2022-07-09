import os

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization

from utils.parser import get_config
from utils.data_loader import DataLoader
from models.bert import BertModel

config = get_config("./configs/config.yaml")
config_train = config.TRAIN
config_dataset = config.DATASET

dataloader = DataLoader()
labels = open(config_dataset.LABELS).read().split('\n')
x_train, y_train = dataloader.load_from_folder(config_dataset.TRAIN_PATH, labels=labels)
x_test, y_test = dataloader.load_from_folder(config_dataset.TEST_PATH, labels=labels)

bert = BertModel(bert_model_name=config_train.MODEL_NAME)
model = bert.build_model()
optimizer = optimization.create_optimizer(init_lr=config_train.LEARNING_RATE,
                                          num_train_steps=len(x_train) * config_train.NUM_EPOCHS,
                                          num_warmup_steps=int(0.1*(len(x_train) * config_train.NUM_EPOCHS)),
                                          optimizer_type='adamw')
loss_method = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")
model.compile(optimizer=optimizer,loss=loss_method,metrics=metric)
model.load_weights("./output/Bert_checkpoint.ckpt")

text = open("Bia6.txt").read()
result = model.predict([text])
print(result)