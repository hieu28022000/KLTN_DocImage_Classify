import os

import numpy as np
import pandas as pd
import pytesseract

from datasets import Dataset
from PIL import Image, ImageDraw, ImageFont



def visualize_bbox(image, ocr_df):
    coordinates = ocr_df[['left', 'top', 'width', 'height']]
    actual_boxes = []
    for idx, row in coordinates.iterrows():
        x, y, w, h = tuple(row) # the row comes in (left, top, width, height) format
        actual_box = [x, y, x+w, y+h] # we turn it into (left, top, left+width, top+height) to get the actual box 
        actual_boxes.append(actual_box)

    draw = ImageDraw.Draw(image, "RGB")
    for box in actual_boxes:
        draw.rectangle(box, outline='red')

    image = image.save("Visualizebbox.jpg")

def pytesseract_ocr(image_path, visualize=False):
    image = Image.open(image_path)
    image = image.convert("RGB")

    ocr_df = pytesseract.image_to_data(image, output_type='data.frame')
    ocr_df = ocr_df.dropna().reset_index(drop=True)
    float_cols = ocr_df.select_dtypes('float').columns
    ocr_df[float_cols] = ocr_df[float_cols].round(0).astype(int)
    ocr_df = ocr_df.replace(r'^\s*$', np.nan, regex=True)
    words = ' '.join([word for word in ocr_df.text if str(word) != 'nan'])
    if visualize:
        visualize_bbox(image, ocr_df)

    return ocr_df

def get_label(dataset_path):
    labels = [label for label in os.listdir(dataset_path)]
    idx2label = {v: k for v, k in enumerate(labels)}
    label2idx = {k: v for v, k in enumerate(labels)}
    return idx2label, label2idx

def create_label_frame(dataset_path):
    images = []
    labels = []

    for label_folder, _, file_names in os.walk(dataset_path):
        if label_folder != dataset_path:
            label = label_folder.split("/")[-1]
            for _, _, image_names in os.walk(label_folder):
                relative_image_names = []
            for image in image_names:
                relative_image_names.append(os.path.join(dataset_path, label , image))
            images.extend(relative_image_names)
            labels.extend([label] * len (relative_image_names)) 

    data = pd.DataFrame.from_dict({'image_path': images, 'label': labels})
    data.head()
    return data

def normalize_box(box, width, height):
    return [
        int(1000 * (box[0] / width)),
        int(1000 * (box[1] / height)),
        int(1000 * (box[2] / width)),
        int(1000 * (box[3] / height)),
    ]

def apply_ocr(example):
        # get the image
        image = Image.open(example['image_path'])

        width, height = image.size
        
        # apply ocr to the image 
        ocr_df = pytesseract.image_to_data(image, output_type='data.frame')
        float_cols = ocr_df.select_dtypes('float').columns
        ocr_df = ocr_df.dropna().reset_index(drop=True)
        ocr_df[float_cols] = ocr_df[float_cols].round(0).astype(int)
        ocr_df = ocr_df.replace(r'^\s*$', np.nan, regex=True)
        ocr_df = ocr_df.dropna().reset_index(drop=True)

        # get the words and actual (unnormalized) bounding boxes
        #words = [word for word in ocr_df.text if str(word) != 'nan'])
        words = list(ocr_df.text)
        words = [str(w) for w in words]
        coordinates = ocr_df[['left', 'top', 'width', 'height']]
        actual_boxes = []
        for idx, row in coordinates.iterrows():
            x, y, w, h = tuple(row) # the row comes in (left, top, width, height) format
            actual_box = [x, y, x+w, y+h] # we turn it into (left, top, left+width, top+height) to get the actual box 
            actual_boxes.append(actual_box)
        
        # normalize the bounding boxes
        boxes = []
        for box in actual_boxes:
            boxes.append(normalize_box(box, width, height))
        
        # add as extra columns 
        assert len(words) == len(boxes)
        example['words'] = words
        example['bbox'] = boxes
        return example

# if __name__ == "__main__":
#     image_path = "./data/dataset/test/01bia/01_18.jpg"
#     ocr_df = pytesseract_ocr(image_path, visualize=True)

#     dataset_path = "./data/dataset/strain"
#     idx2label, label2idx = get_label(dataset_path)
#     data = create_label_frame(dataset_path)
#     dataset = Dataset.from_pandas(data)
#     updated_dataset = dataset.map(apply_ocr)
#     print(updated_dataset)
#     df = pd.DataFrame.from_dict(updated_dataset)
#     print(len(df["words"][11]))
#     print(df["words"][11])