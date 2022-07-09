
import os 

import tensorflow as tf
import tensorflow_addons as tfa

from .utils.parser import get_config
from .models.vision_transformer import vision_transformer_model


def create_model(config_path):
    # Get config
    config = get_config(config_path)
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

    # Create loss and optimizer
    optimizer = tfa.optimizers.AdamW(learning_rate=config_train.LEARNING_RATE, weight_decay=config_train.WEIGHT_DECAY)
    loss_method = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric1 = tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")
    metric2 = tf.keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy")

    # Compile
    model.compile(optimizer=optimizer, loss=loss_method, metrics=[metric1, metric2])

    return model

# Predict
vit_model = create_model("./models/vision_transformer/configs/vision_transformer.yaml")

# Training
# vit_model = create_model("./configs/vision_transformer.yaml")