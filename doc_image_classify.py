import os
import glob
import logging
import math
import time

import cv2
import numpy as np
import pandas as pd
import torch
import tensorflow as tf
from datasets import Dataset
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support

from utils.parser import get_config
from models.vision_transformer.vision_transformer import vit_model
from models.bert.bert import bert_model
from models.layoutlm.models.layoutlm import layoutlm_classify_model
from models.layoutlm.utils.dataloader import apply_ocr, create_label_frame
from models.layoutlm.models.encoding import create_features, encode_example


CONFIG = get_config("./configs/doc_image_classify.yaml")


class DocImageClassify(object):
    def __init__(self, model_name):
        if model_name == "resnet50":
            self.model_name = model_name
            self.model = tf.keras.models.load_model(CONFIG.RESNET50_MODEL_PATH)
        
        elif model_name == "vgg16":
            self.model_name = model_name
            self.model = tf.keras.models.load_model(CONFIG.VSGG16_MODEL_PATH)
        
        elif model_name == "xception":
            self.model_name = model_name
            self.model = tf.keras.models.load_model(CONFIG.XCEPTION_MODEL_PATH)
        
        elif model_name == "vision_transformer":
            self.model_name = model_name
            self.model = vit_model
            self.model.load_weights(CONFIG.VISION_TRANSFORMER_WEIGHT_PATH)
        
        elif model_name == "bert":
            self.model_name = model_name
            self.model = bert_model
            self.model.load_weights(CONFIG.BERT_WEIGHT_PATH)
        
        elif model_name == "layoutlm":
            self.model_name = model_name
            self.model = layoutlm_classify_model(CONFIG.LAYOUTLM_MODEL_PATH)
        
        else:
            raise Exception("MODEL NAME NOT FOUND")

    def __softmax(self, vector):
        softmax_vector = []
        sum_exp = 0 
        for e in vector:
            sum_exp += math.exp(e)
        for e in vector:
            softmax_vector.append(math.exp(e)/sum_exp)
        return softmax_vector

    def vatf_preprocess(self, image):
        image = cv2.resize(image, dsize=(331,468), interpolation=cv2.INTER_AREA)
        image = np.asarray(image)/255
        image = np.array([image])
        return image

    def vit_preprocess(self, image):
        image = cv2.resize(image, dsize=(32,32), interpolation=cv2.INTER_AREA)
        image = np.array([image])
        return image

    def bert_preprocess(self, text):
        text = np.array([text])
        return text

    def layoutlm_preprocess(self, image_path):
        test_data = pd.DataFrame.from_dict({'image_path': [image_path]})
        test_dataset = Dataset.from_pandas(test_data)
        updated_dataset = test_dataset.map(apply_ocr)
        encoded_dataset = updated_dataset.map(lambda example: encode_example(example=example), features=create_features(label=False))
        encoded_dataset.set_format(type='torch', columns=['input_ids', 'bbox', 'attention_mask', 'token_type_ids'])
        test_dataloader = torch.utils.data.DataLoader(encoded_dataset, batch_size=1, shuffle=True)
        test_batch = next(iter(test_dataloader))
        input_ids = test_batch["input_ids"].to(torch.device("cpu"))
        bbox = test_batch["bbox"].to(torch.device("cpu"))
        attention_mask = test_batch["attention_mask"].to(torch.device("cpu"))
        token_type_ids = test_batch["token_type_ids"].to(torch.device("cpu"))

        return input_ids, bbox, attention_mask, token_type_ids

    def layoutlm_correc_cls(self, cls):
        if cls == 0:
            return 3
        elif cls == 1:
            return 0
        elif cls == 2:
            return 5
        elif cls == 3:
            return 6
        elif cls == 4:
            return 2
        elif cls == 5:
            return 4
        elif cls == 6:
            return 1
        else:
            return 0

    def predict(self, file_path):
        if self.model_name in ["resnet50", "vgg16", "xception"]:
            image = self.vatf_preprocess(file_path)
            result = self.model(image)[0]
            result = np.array(result)
            return np.argmax(result), np.amax(result)

        if self.model_name == "vision_transformer":
            image = self.vit_preprocess(file_path)
            result = self.model.predict(image)[0]
            result = self.__softmax(result)
            result = np.array(result)
            return np.argmax(result), np.amax(result)

        if self.model_name == "bert":
            text = self.bert_preprocess(file_path)
            result = self.model.predict(text)[0]
            result = self.__softmax(result)
            result = np.array(result)
            return np.argmax(result), np.amax(result)

        if self.model_name == "layoutlm":
            input_ids, bbox, attention_mask, token_type_ids = self.layoutlm_preprocess(file_path)
            result = self.model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids)
            result = self.__softmax(result.logits[0])
            result = np.array(result)
            return self.layoutlm_correc_cls(np.argmax(result)), np.amax(result)
    
    def evaluate(self, data_path):
        labels = open(CONFIG.LABELS_PATH).read().split('\n')
        predict_list = []
        gr_truth_list = []
        time_list = []
        true_pred = 0
        total = 0

        for file_path in tqdm(glob.glob(os.path.join(data_path, "*/*")), desc="Evaluate"):
            gr_truth = labels.index(file_path.split('/')[-2])
            start = time.time()
            pred_cls = self.predict(file_path)[0]
            time_list.append(time.time() - start)
            gr_truth_list.append(gr_truth)
            predict_list.append(pred_cls)
            if gr_truth == pred_cls:
                true_pred += 1
            total +=1
        
        (precision, recall, f1_score, _) = precision_recall_fscore_support(gr_truth_list, predict_list, average='macro')
        average_time = np.average(time_list)
        acc = true_pred/total
        print('average_time',average_time)
        print('acc', acc)
        print('precision, recall, f1_score', precision, recall, f1_score)
        eva_dict = {
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1_score,
            "Accuracy": acc,
            "Average_runtime": average_time
        }
        return eva_dict


# if __name__ == "__main__":
#     model_name = "layoutlm"
#     model = DocImageClassify(model_name=model_name)
    
#     predict_cls, score = model.predict('./test.jpg')
#     print(predict_cls, score)

#     test_data_path = "./data/images/test/"
#     print('\n\n',model_name)
#     eva_dict = model.evaluate(data_path=test_data_path)
