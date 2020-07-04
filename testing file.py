# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 16:38:17 2020

@author: Anushka
"""


from tensorflow.keras.models import load_model
model=load_model('mymodel.h5')

loaded=CountVectorizer(decode_error="replace",vocabulary=pickle.load(open("sentiment.pkl", "rb")))
da="stocks are good"
da= da.split("delimiter")
d=loaded.transform(da)
result=model.predict(d)
print(result)
prediction=result>0.5
print(prediction)

if (prediction==True):
    print("Positive")
else:
    print("Negative")
