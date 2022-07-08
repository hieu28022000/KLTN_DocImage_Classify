
import os
import argparse
import logging

import tensorflow as tf

from utils.parser import get_config
from utils.dataloader import DataLoader
from vision_transformer import create_model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="./configs/vision_transformer.yaml",
                        help="Path to config file")
    parser.add_argument("--model_saved_path", type=str, default="./output",
                        help="Path to save model")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to resume model")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Get config
    config = get_config(args.config_path)
    config_train = config.TRAIN
    config_dataset = config.DATASET

    # Load device
    if config_train.DEVICE == "gpu":
        config_train.DEVICE = "/gpu" + ":" + config_train.GPU_ID

    with tf.device(config_train.DEVICE):
        # Load data
        logging.info("Loading dataset")
        dataloader = DataLoader()
        labels = open(config_dataset.LABELS).read().split('\n')
        x_train, y_train = dataloader.load_from_folder(config_dataset.TRAIN_PATH, labels=labels, image_size=[config_train.IMAGE_SIZE,config_train.IMAGE_SIZE])
        x_test, y_test = dataloader.load_from_folder(config_dataset.TEST_PATH, labels=labels, image_size=[config_train.IMAGE_SIZE,config_train.IMAGE_SIZE])

        model = create_model(args.config_path)
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            os.path.join(args.model_saved_path, 'ViT_checkpoint.ckpt'),
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=True,
        )
        
        # Resume
        if args.resume:
            model.load_weights(args.resume)

        logging.info("Training...")
        model.fit(
            x=x_train,
            y=y_train,
            batch_size=config_train.BATCH_SIZE,
            epochs=config_train.NUM_EPOCHS,
            validation_data=(x_test, y_test),
            callbacks=[checkpoint_callback],
        )

        model.load_weights(os.path.join(args.model_saved_path,'ViT_checkpoint.ckpt'))
        _, accuracy, top_5_accuracy = model.evaluate(x_test, y_test)
        print(f"Test accuracy: {round(accuracy * 100, 2)}%")
        print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")