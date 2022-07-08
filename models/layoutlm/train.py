import os
import logging
import argparse

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

import torch
import pandas as pd
from datasets import Dataset
from transformers import AdamW
from tqdm import tqdm

from utils.parser import get_config
from utils.dataloader import get_label, create_label_frame, apply_ocr
from models.encoding import create_features, encode_example
from models.layoutlm import layoutlm_classify_model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="./configs/layoutlm.yaml",
                        help="Path to config file")
    parser.add_argument("--model_saved_path", type=str, default="./output/",
                        help="Path to save model")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to resume model")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Load config
    configs = get_config(args.config_path)
    config_train = configs.TRAIN
    config_test = configs.TEST
    config_dataset = configs.DATASET
    logging.info("Loading config from %s", args.config_path)

    # Load device
    if config_train.DEVICE == "gpu":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    
    # Load train data
    idx2label, label2idx = get_label(config_dataset.TRAIN_PATH)
    data = create_label_frame(dataset_path=config_dataset.TRAIN_PATH)
    dataset = Dataset.from_pandas(data)
    updated_dataset = dataset.map(apply_ocr)

    encoded_dataset = updated_dataset.map(lambda example: encode_example(example=example, label2idx=label2idx), features=create_features())
    encoded_dataset.set_format(type='torch', columns=['input_ids', 'bbox', 'attention_mask', 'token_type_ids', 'label'])
    dataloader = torch.utils.data.DataLoader(encoded_dataset, batch_size=1, shuffle=True)
    batch = next(iter(dataloader))

    # Load test data
    test_data = create_label_frame(dataset_path=config_dataset.TEST_PATH)
    test_dataset = Dataset.from_pandas(test_data)
    updated_test_dataset = test_dataset.map(apply_ocr)

    encoded_test_dataset = updated_test_dataset.map(lambda example: encode_example(example=example, label2idx=label2idx), features=create_features())
    encoded_test_dataset.set_format(type='torch', columns=['input_ids', 'bbox', 'attention_mask', 'token_type_ids', 'label'])
    test_dataloader = torch.utils.data.DataLoader(encoded_test_dataset, batch_size=1, shuffle=False)
    test_batch = next(iter(test_dataloader))

    # Model
    if args.resume:
        model = layoutlm_classify_model(args.resume, len(label2idx))
    else:
        model = layoutlm_classify_model(config_train.PRETRAIN_MODEL, len(label2idx))

    model.to(device)

    # Create optimizer and loss
    learning_rate = config_train.LR
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Training
    best_acc = 0.
    global_step = 0
    # t_total = len(dataloader) * config_train.NUM_EPOCHS
    
    logging.info("Training %s epoch is starting", config_train.NUM_EPOCHS)
    model.train()
    for epoch in range(1, config_train.NUM_EPOCHS + 1):
        if epoch-1 in config_train.ADJUST_LR_AFTER:
            learning_rate = learning_rate * 0.1
            optimizer = AdamW(model.parameters(), lr=learning_rate)
        running_loss = 0.
        true = 0

        # Train loop
        for batch in tqdm(dataloader, desc='epoch '+str(epoch)):
            input_ids = batch["input_ids"].to(device)
            bbox = batch["bbox"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["label"].to(device)
            outputs = model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
            loss = outputs.loss
            running_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
        
        # Evaluate
        for batch in tqdm(test_dataloader, desc="evaluate"):
            input_ids = batch["input_ids"].to(device)
            bbox = batch["bbox"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["label"].to(device)
            result = model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids)
            predictions = result.logits.argmax(-1)
            if int(predictions[0]) == int(labels[0]):
                true +=1
            
        acc = true/len(test_data)
        logging.info("Epoch %s - Learning rate %s - Accuracy %s", int(epoch), str(learning_rate), str(acc))
        if acc > best_acc:
            best_acc = acc
            model.save_pretrained(os.path.join(args.model_saved_path, "best_model"))
            logging.info("Best model saved at %s", args.model_saved_path)
        if epoch % config_train.SAVE_MODEL_AFTER_EACH == 0:
            model.save_pretrained(os.path.join(args.model_saved_path, str(epoch)))
            logging.info("Model at epoch %s saved %s", str(epoch), args.model_saved_path)

