from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
import json
import pickle


from tokenizers import BertWordPieceTokenizer
import torch 
import torch.nn as nn
import torch.nn.functional as F
from AttentionTransformer import ClassificationTransformer
from AttentionTransformer.utilities import count_model_parameters, device

from predict import getTextPrediction


app = Flask(__name__)
CORS(app)


global tokenizer, model

tokenizer = BertWordPieceTokenizer('../../models/bert-base-cased-vocab.txt', lowercase = True)
model = ClassificationTransformer(28996, 2, 200, 0)
model = model.to(device())
model = torch.load('../../models/classification_transformer_training_epoch_57.pt', map_location = device())
model = model.eval()




@app.route('/api/predict_text', methods = ['POST'])
def predict_text():
    req = request.get_json()
    text = req['text']
    prediction = getTextPrediction(text, model, tokenizer)

    return jsonify({"text": text, "result": prediction})



if __name__ == '__main__':
    app.run(debug = True)