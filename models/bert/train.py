
import argparse
import logging

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

import tensorflow as tf

from utils.parser import get_config
from utils.data_loader import DataLoader
from bert import create_model
import matplotlib.pyplot as plt

tf.get_logger().setLevel('ERROR')   

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="./configs/bert.yaml",
                        help="Path to config file")
    parser.add_argument("--model_saved_path", type=str, default="./output/Bert_checkpoint.ckpt",
                        help="Path to save model")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to resume model")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Load config
    configs = get_config(args.config_path)
    config_train = configs.TRAIN
    config_dataset = configs.DATASET

    # Load device
    if config_train.DEVICE == "gpu":
        config_train.DEVICE = "/gpu" + ":" + config_train.GPU_ID

    with tf.device(config_train.DEVICE):
        # Load data
        dataloader = DataLoader()
        labels = open(config_dataset.LABELS).read().split('\n')
        x_train, y_train = dataloader.load_from_folder(config_dataset.TRAIN_PATH, labels=labels)
        x_test, y_test = dataloader.load_from_folder(config_dataset.TEST_PATH, labels=labels)

        # Training
        model = create_model(args.config_path)
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            args.model_saved_path,
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=True,
        )

        if args.resume:
            model.load_weights(args.resume)

        logging.info("Training...")
        history = model.fit(
            x=x_train,
            y=y_train,
            batch_size=config_train.BATCH_SIZE,
            epochs=config_train.NUM_EPOCHS,
            validation_data=(x_test, y_test),
            callbacks=[checkpoint_callback],
        )

        model.load_weights(args.model_saved_path)
        _, accuracy = model.evaluate(x_test, y_test)
        print(f"Test accuracy: {round(accuracy * 100, 2)}%")