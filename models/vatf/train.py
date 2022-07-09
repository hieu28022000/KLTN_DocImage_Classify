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
import logging
import os

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

import tensorflow as tf
from tqdm import tqdm

from utils.data_loader import Dataset
from models.resnet50_backbone import ResNet50Classify
from models.vgg16_backbone import VGG16Classify
from models.xception_backbone import XceptionClassify
from utils import get_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="./configs/vatf.yaml",
                        help="Path to config file")
    parser.add_argument("--model_saved_path", type=str, default="./output/",
                        help="Path to save model")
    parser.add_argument("--resume", type=str,
                        help="Resume training model from this path")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Load config
    configs = get_config(args.config_path)
    config_train = configs.TRAIN
    config_test = configs.TEST
    config_dataset = configs.DATASET

    # Load device
    if config_train.DEVICE == "gpu":
        config_train.DEVICE = "/gpu" + ":" + config_train.GPU_ID

    with tf.device(config_train.DEVICE):
        # Resume model
        if args.resume is not None:
            model = tf.keras.models.load_model(args.resume)
        else:
            model = ResNet50Classify()
            # model = VGG16Classify()
            # model = XceptionClassify()

        # Load data
        labels = open(config_dataset.LABELS_PATH).read().split('\n')
        image_size = (int(config_dataset.IMAGE_SIZE.split(',')[0]), int(config_dataset.IMAGE_SIZE.split(',')[1]))
        train_dataset = Dataset(labels=labels, \
                                dataset_path=config_dataset.TRAIN_FOLDER, \
                                image_size=image_size)
        test_dataset = Dataset(labels=labels, \
                                dataset_path=config_dataset.TEST_FOLDER, \
                                image_size=image_size)

        # Create loss and optimizer
        optimizer = tf.keras.optimizers.SGD(learning_rate=config_train.LR, momentum=config_train.MOMENTUM)
        loss_method = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

        # Training
        best_acc = 0.
        for epoch in range(1, config_train.NUM_EPOCHS + 1):
            pb = tf.keras.utils.Progbar(len(train_dataset), stateful_metrics=['epoch', 'learning rate', 'loss',])
            if epoch-1 in config_train.ADJUST_LR_AFTER:
                lr = optimizer.learning_rate
                optimizer.learning_rate.assign(lr * 0.1)
            
            # Train loop
            for train_point in train_dataset:
                with tf.GradientTape() as tape:
                    predict_score = model(train_point[0])
                    loss = loss_method([train_point[1]], predict_score[0])
                    gradients = tape.gradient(loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                    values=[('epoch', int(epoch)),('learning rate', optimizer.learning_rate.numpy()), ('loss', loss)]
                    pb.add(1, values)

            # Evaluate
            true_pred = -1
            for test_point in tqdm(test_dataset, desc = 'Evaluate'):
                predict_score = model(test_point[0])
                pred_cls = tf.argmax(predict_score[0])
                if pred_cls == test_point[-1]:
                    true_pred += 1
            acc = true_pred/len(test_dataset)
            logging.info("Epoch %s - Learning rate %s - Loss %s - Accuracy %s", int(epoch), str(optimizer.learning_rate.numpy()), str(loss.numpy()), acc)
            
            if acc > best_acc:
                best_acc = acc
                model.save(os.path.join(args.model_saved_path, 'best_model'))
            if epoch % config_train.SAVE_MODEL_AFTER_EACH == 0:
                model.save(os.path.join(args.model_saved_path, str(epoch)))