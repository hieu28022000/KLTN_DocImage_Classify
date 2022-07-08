
import os
import glob
import random
import numpy as np

class DataLoader(object):
    def __init__(self):
        pass

    def load_from_folder(self, dataset_path, labels):
        dataset = []
        for file_text in glob.glob(os.path.join(dataset_path, "*/*")):
            text = open(file_text).read()
            text_cls = labels.index(file_text.split("/")[-2])
            dataset.append([text, text_cls])
        random.shuffle(dataset)
        datas = [point[0] for point in dataset]
        clses = [[point[1]] for point in dataset]
        return np.array(datas), np.array(clses)