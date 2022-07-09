"""
Description       : 
Author            : NGUYEN QUANG HIEU
Maintainer        : NGUYEN QUANG HIEU
Date              : 04/03/2022 
Version           : 
Usage             :
Notes             :
"""

import glob
import os
import random

import numpy as np
from PIL import Image


class Dataset(object):
    def __init__(self, labels, dataset_path, image_size=(256, 256), shuffle=True):
        
        self.image_size = image_size
        self.labels = labels

        self.data_points = []
        
        for image_path in glob.glob(os.path.join(dataset_path, "*/*")):
            if image_path.split('.')[-1] not in ["jpg", "png", "jpeg"]:
                continue
            else:
                label = image_path.split('/')[-2]
                if label not in self.labels:
                    raise Exception("Class folder not in labels list")
                
                self.data_points.append([image_path, labels.index(label)])
        if shuffle:
            random.shuffle(self.data_points)

    def __len__(self):
        return len(self.data_points)

    def __preprocess(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image = image.resize(self.image_size, Image.ANTIALIAS)
        image = np.asarray(image)/255
        image = np.array([image])
        return image

    def __getitem__(self, index):
        image_path = self.data_points[index][0]
        image_cls = self.data_points[index][1]
        data_point = [self.__preprocess(image_path), image_cls]
        return data_point


# if __name__ == "__main__":
#     labels = ["basketball", "football", "tenis"]
#     dataset_path = './data/train/'

#     dataset = Dataset(labels, dataset_path)
#     first_data_point = dataset[0]
#     image_shape = first_data_point[0].shape
#     image_cls = first_data_point[1]
#     print(image_shape, image_cls)
