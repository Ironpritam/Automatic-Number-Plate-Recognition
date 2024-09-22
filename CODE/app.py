import os
import sys

from check2 import Preprocessing
#from check3 import get_text

import detect
import argparse
import sys
from pathlib import Path




# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Some utilites
import numpy as np
from util import base64_to_pil


# Declare a flask app
app = Flask(__name__)


# You can use pretrained model from Keras
# Check https://keras.io/applications/
# or https://www.tensorflow.org/api_docs/python/tf/keras/applications

#from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
#model = MobileNetV2(weights='imagenet')

#print('Model loaded. Check http://127.0.0.1:5000/')


# Model saved with Keras model.save()
#MODEL_PATH = './your_model.pt'

# Load your own trained model
# model = load_model(MODEL_PATH)
# model._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')


#def model_predict(img, model):
#    img = img.resize((224, 224))

    # Preprocessing the image
#    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
#    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
#    x = preprocess_input(x, mode='tf')

#    preds = model.predict(x)
#    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('newin.html')

@app.route('/english_detection', methods=['GET'])
def english_detection():
    return render_template('index.html')

@app.route('/marathi_detection', methods=['GET'])
def marathi_detection():
    return render_template('index_for_marathi.html')

@app.route('/about', methods=['GET'])
def about():
    return render_template('about.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request
        img = base64_to_pil(request.json)

        # Save the image to ./uploads
        img.save("./uploads/image.png")
        result = detect.run()

        # Make prediction
        #preds = model_predict(img, model)

        # Process your result for human
        #pred_proba = "{:.3f}".format(np.amax(preds))    # Max probability
        #pred_class = decode_predictions(preds, top=1)   # ImageNet Decode

        #result = str(pred_class[0][0][1])               # Convert to string
        #result = result.replace('_', ' ').capitalize()

        # Serialize the result, you can add additional fields

        return jsonify(result = str(result))

    #return "Succesfull"


@app.route('/predict_marathi', methods=['GET', 'POST'])
def predict_marathi():
    if request.method == 'POST':
        # Get the image from post request
        img = base64_to_pil(request.json)

        # Save the image to ./uploads
        img.save("./uploads/image.png")
        result = detect.run(lang='mr')

        return jsonify(result = str(result))


if __name__ == '__main__':
    # app.run(port=5002, threaded=False)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
