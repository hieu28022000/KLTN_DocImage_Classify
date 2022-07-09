import os
import glob
import time

import cv2
import imutils
import numpy as np
from tqdm import tqdm

from .soc.core.text_detection.printed_vie_detection.predict_model import DBPredict as PVIE_DET
from .soc.core.text_recognition.printed_vie_recognition.predict_model import CRNNPredict as PVIE_REG

vie_det = PVIE_DET()
vie_rec = PVIE_REG()

def sort_boxes(boxes, lsimg):
    y_centers = []
    for box in boxes:
        y_cen = (box[0][1] + box[1][1] + box[2][1] + box[3][1])/4
        y_centers.append(y_cen)
    
    result_boxes = []
    result_lsimg = []
    maxx = np.amax(y_centers) + 1
    for i in range(len(y_centers)):
        index = np.argmin(y_centers)
        y_centers[index] = maxx
        result_boxes.append(boxes[index])
        result_lsimg.append(lsimg[index])
        
    return result_boxes, result_lsimg

def ocr(image):
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), -1, 1.0)
    image = cv2.warpAffine(image, M, (w, h))
    # cv2.imwrite("temp1.jpg", image)

    boxes, lsimg = vie_det.predict(image)
    boxes, lsimg = sort_boxes(boxes, lsimg)

    txts = vie_rec.predict(lsimg)
    text = ' '.join(txts)
    return text


if __name__ == "__main__":
    
    print(ocr(cv2.imread('./../../test.jpg')))

    # time_list = []

    # for image_path in tqdm(glob.glob("./../../data/images/s*/*/*")):
    #     start = time.time()
    #     try:
    #         image = cv2.imread(image_path)
    #         text = ocr(image)

    #         save_file = image_path.replace("images", "texts")[:-4] + '.txt'
    #         with open(save_file,'w') as f:
    #             f.write(text)
    #     except:
    #         pass
    #     time_list.append(time.time() - start)
    # print(np.average(time_list))
        