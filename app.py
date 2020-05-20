# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 11:29:19 2019

@author: lalit
"""
    
from flask import render_template, Flask, request,url_for
from keras.models import load_model
import pickle 
from keras.preprocessing import sequence
import tensorflow as tf
graph = tf.get_default_graph()
#with open(r'CountVectorizer','rb') as file:
 #   cv=pickle.load(file)
cv=pickle.load(open('cv_tranform.pkl','rb'))
cla = load_model('NLPmodel.h5')
#cla.compile(optimizer='adam',loss='binary_crossentropy')
app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict():
	if request.method == 'POST':
		topic = request.form['Review']
		data = [topic]
		vect = cv.transform(data).toarray()
		vect = sequence.pad_sequences(vect, maxlen=1500)
		with graph.as_default():
			my_prediction = cla.predict_classes(vect)
		#my_prediction = cla.predict(vect)
	return render_template('result.html',prediction = my_prediction)


        
        
        
        
       


if __name__ == '__main__':
    app.run( debug = True)
    
