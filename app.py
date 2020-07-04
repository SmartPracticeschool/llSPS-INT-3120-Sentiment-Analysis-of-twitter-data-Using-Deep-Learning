import numpy as np
import pickle
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from  sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.models import load_model
model=load_model('mymodel.h5')

loaded=CountVectorizer(decode_error="replace",vocabulary=pickle.load(open("sentiment.pkl", "rb")))
app=Flask(__name__,template_folder='templates')

@app.route('/')
def home():
    return render_template('login.html')
@app.route('/index',methods=['POST'])
def index():
    return render_template('index.html')
@app.route('/y_predict',methods=['POST'])
def y_predict():
    x_test=request.form['text1']
    x_test= x_test.split("delimiter")
    print(x_test)
    result=model.predict(loaded.transform(x_test))
    print(result)
    prediction=result>0.5
    print(prediction)
    output=result[0][0]
    if (output>0.5):
        op="Positive"
    else:
        op="Negative"
    print(op)
    return render_template('index.html', prediction_text='Sentiment of your tweet is : {}'.format(op))
    
if __name__ == "__main__":
    app.run(debug=True)
    

