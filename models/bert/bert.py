import os
import glob

import tensorflow as tf
from official.nlp import optimization

from .utils.parser import get_config
from .models.bert import BertModel

def create_model(config_path):
    config = get_config(config_path)
    bert = BertModel(bert_model_name=config.TRAIN.MODEL_NAME)
    model = bert.build_model()
    numdata_point = len([name for name in glob.glob(os.path.join(config.DATASET.TRAIN_PATH, "*/*"))])
    optimizer = optimization.create_optimizer(init_lr=config.TRAIN.LEARNING_RATE,
                                        num_train_steps=numdata_point * config.TRAIN.NUM_EPOCHS,
                                        num_warmup_steps=int(0.1*(numdata_point * config.TRAIN.NUM_EPOCHS)),
                                        optimizer_type='adamw')
    loss_method = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")

    # Training
    model.compile(optimizer=optimizer,loss=loss_method,metrics=metric)
    
    return model

# Predict
bert_model = create_model("./models/bert/configs/bert.yaml")

# Training
# bert_model = create_model("./configs/bert.yaml")