
import os 
import glob
import random

import cv2
import numpy as np
from tqdm import tqdm

class DataLoader(object):
    def __init__(self):
        pass

    def load_from_folder(self, dataset_path, labels, image_size):
        dataset = []
        for image_path in tqdm(glob.glob(os.path.join(dataset_path, "*/*"))):
            image = cv2.imread(image_path)
            image = cv2.resize(image, (image_size[0],image_size[1]), interpolation = cv2.INTER_AREA)
            image_cls = labels.index(image_path.split('/')[-2])
            dataset.append([image, image_cls])
        random.shuffle(dataset)
        datas = [point[0] for point in dataset]
        clses = [[point[1]] for point in dataset]
        return np.array(datas), np.array(clses)