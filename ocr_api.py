
import cv2
import numpy as np
from flask import Flask, request, jsonify, make_response

from configs.api_config import api_config
from models.bert.ocr import ocr

app = Flask(__name__)
app.config['DEBUG'] = api_config['debug']


def image_loader(filestr):
    npimg = np.fromstring(filestr, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    return img

def backend(image):
    h, w, c = image.shape
    image = cv2.resize(image, (700, int(h*700/w)), interpolation = cv2.INTER_AREA)
    text = ocr(image)
    return text


@app.route('/ocr/alive', methods=['POST', 'GET'])
def alive():
    return 'OCR document images is running...'

@app.route('/ocr', methods=["POST"])
def post():
    try:
        image = image_loader(request.files['image'].read())
        text = backend(image)
        result = {
            'text': str(text)
        }
    except:
        result = {
            'text': ''
        }
    return make_response(jsonify(result))

app.run(host=api_config['host'], port=api_config['port'])
