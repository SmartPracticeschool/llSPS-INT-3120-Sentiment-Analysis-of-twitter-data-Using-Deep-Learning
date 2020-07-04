import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset= pd.read_csv('dataset/datasets_695933_1217821_stock_data.csv')
dataset['Sentiment']=dataset['Sentiment'].replace(-1,0)
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps= PorterStemmer()
data=[]
for i in range(0,5791):
    review=re.sub('[^a-zA-Z]',' ',dataset['Text'][i])
    review =review.lower()
    review= review.split()
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    data.append(review)
from  sklearn.feature_extraction.text import CountVectorizer
cv =CountVectorizer(max_features=5000)
X=cv.fit_transform(data).toarray()
y= dataset.iloc[:,-1].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

import pickle
pickle.dump(cv.vocabulary_,open("sentiment.pkl","wb"))
import keras
from tensorflow.keras.models import Sequential #is used to initalize the model
from tensorflow.keras.layers import Dense #is used to build layers


model=Sequential()
model.add(Dense(input_dim=5000,kernel_initializer="random_uniform",activation="sigmoid",units=2000))
model.add(Dense(kernel_initializer="random_uniform",activation="sigmoid",units=200))
model.add(Dense(units=1,kernel_initializer='random_uniform',activation='sigmoid'))
model.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy'])
model.fit(x_train,y_train,epochs=80,batch_size=32)

print(x_test.shape)
model.save('mymodel.h5')


y_pred=model.predict(x_test)
y_pred=(y_pred>0.5)
y_pred
#accuracy
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)

