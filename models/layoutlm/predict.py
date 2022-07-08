import os 


import torch
import pandas as pd
from datasets import Dataset

from models.layoutlm import layoutlm_classify_model
from utils.parser import get_config
from utils.dataloader import apply_ocr
from models.encoding import create_features, encode_example

PRETRAIN_MODEL = "./output/best_model"
config_path = "./configs/layoutlm.yaml"

config = get_config(config_path)
label = open(config.DATASET.LABELS_PATH).read().split('\n')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = layoutlm_classify_model(PRETRAIN_MODEL, len(label))
model.to(device)

image_path = './test/test.jpg'
test_data = pd.DataFrame.from_dict({'image_path': [image_path]})

test_dataset = Dataset.from_pandas(test_data)
updated_dataset = test_dataset.map(apply_ocr)

encoded_dataset = updated_dataset.map(lambda example: encode_example(example=example), features=create_features(label=False))
encoded_dataset.set_format(type='torch', columns=['input_ids', 'bbox', 'attention_mask', 'token_type_ids'])

test_dataloader = torch.utils.data.DataLoader(encoded_dataset, batch_size=1, shuffle=True)
test_batch = next(iter(test_dataloader))

input_ids = test_batch["input_ids"].to(device)
bbox = test_batch["bbox"].to(device)
attention_mask = test_batch["attention_mask"].to(device)
token_type_ids = test_batch["token_type_ids"].to(device)

outputs = model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids)
classification_logits = outputs.logits
classification_results = torch.softmax(classification_logits, dim=1).tolist()[0]

print(classification_results)
for i in range(len(label)):
    print(f"{label[i]}: {int(round(classification_results[i] * 100))}%")