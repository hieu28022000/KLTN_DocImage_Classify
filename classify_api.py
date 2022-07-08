
import os
import json
import requests

import cv2
import numpy as np
from flask import Flask, request, jsonify, make_response, render_template

from configs.api_config import api_config
from doc_image_classify import DocImageClassify


app = Flask(__name__, template_folder=api_config['template_fol'])
app.config['DEBUG'] = api_config['debug']

vgg16_model = DocImageClassify(model_name='vgg16')
resnet50_model = DocImageClassify(model_name='resnet50')
xception_model = DocImageClassify(model_name='xception')
vision_transformer_model = DocImageClassify(model_name='vision_transformer')
bert_model = DocImageClassify(model_name='bert')
layoutlm_model = DocImageClassify(model_name='layoutlm')

DOCUMENT_TYPE = ["Bìa", "Thông tin quản lý", "Bản cam kết", "Báo cáo kiểm toán", "Bảng cân đối kế toán", "Kết quả hoạt động", "Thuyết minh báo cáo"]

def image_loader(filestr):
    npimg = np.fromstring(filestr, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    return img

def backend(model_name, data):
    if model_name == "vgg16":
        predict_cls, score = vgg16_model.predict(data)
    elif model_name == "resnet50":
        predict_cls, score = resnet50_model.predict(data)
    elif model_name == "xception":
        predict_cls, score = xception_model.predict(data)
    elif model_name == "vision_transformer":
        predict_cls, score = vision_transformer_model.predict(data)
    elif model_name == "bert":
        cv2.imwrite('./temp.jpg', data)
        response = requests.post(url=api_config['url_ocr'], files={'image': open('./temp.jpg', 'rb')})
        data = json.loads(response.text)['text'] 
        predict_cls, score = bert_model.predict(data)
    elif model_name == "layoutlm":
        cv2.imwrite('./temp.jpg', data)
        predict_cls, score = layoutlm_model.predict('./temp.jpg')
        if os.path.exists('./temp.jpg'):
            os.remove('./temp.jpg')
    else:
        predict_cls, score = 0, 0
    return predict_cls, score

@app.route('/home')
def index():
    return render_template("index.html")

@app.route('/classify/alive', methods=['POST', 'GET'])
def alive():
    return 'Document images classify is running...'

@app.route('/classify', methods=["POST"])
def post():
    model_name = request.form['model_name']
    image_input = request.files['image']
    data = image_loader(image_input.read())

    predict_cls, score = backend(model_name, data)
    result = {
        'class': int(predict_cls),
        'label': str(DOCUMENT_TYPE[int(predict_cls)]),
        'score': float(score)
    }
    return make_response(jsonify(result), 200)

app.run(host=api_config['host'], port=api_config['port'])