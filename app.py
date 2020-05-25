# -*- coding: utf-8 -*-
"""
Created on Mon May 25 16:42:41 2020

@author: VISHAL
"""

from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle



app = Flask(__name__)
model = pickle.load(open('Nlp.pkl', 'rb'))
cv=pickle.load(open('transform.pkl','rb'))

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	

	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		vect = cv.transform(data).toarray()
		my_prediction = model.predict(vect)
	return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)