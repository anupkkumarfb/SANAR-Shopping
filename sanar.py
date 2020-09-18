from keras.datasets import imdb

import json
import os
import flask
from flask import Flask,jsonify,request
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
#from flask_cors import CORS
#from predictor import my_bitcoin_predictor
#from flask_ngrok import run_with_ngrok  ## Enable ngrok while using Google Colab
from keras.models import load_model

d = imdb.get_word_index()
max_review_length = 500
model_1 = load_model('DeepLearning_Predict_9182020.h5')

app = Flask(__name__)
#run_with_ngrok(app)    ## Enable ngrok while using Google Colab

@app.route('/')
def index():
    # Main page
    #print("Hello, %s!" % auth.username())
    return ("There is no UI")

@app.route('/predict', methods=['GET', 'POST'])
def predict():

    review = request.args['a']
    rev = review
    print("a in GET: ", review)

    words = review.split()
    review = []
    for word in words:
      if word not in d: 
        #print("Word: ", word)
        review.append(2)
      else:
        #print("Word in else: ", word)
        review.append(d[word]+3) 
        #print(d[word])

    # Required because of a bug in Keras when using tensorflow graph cross threads
    #with graph.as_default():
    print("Above prediction, REview: ", review)
    review = sequence.pad_sequences([review],truncating='pre', padding='pre', maxlen=max_review_length)
    prediction = model_1.predict(review)
    score = prediction[0][0]
    print("Score: ", score * 100)

    if (score > 0.7):
        res = "Excellent experience"
    elif (score < 0.2):
        res = "Worst Experience"
    else:
        res = "Just OK"
        
    print("Final Result: ", res)
    #res = "Sample"
        
    data = {'score': int(score * 100), 
            'feedback' : res}
    return flask.jsonify(data)

if __name__ == "__main__":
    #max_review_length = 500
    #model_1 = load_model('DeepLearning_Predict_9182020.h5')
    #model_1.summary()
    app.run() 
