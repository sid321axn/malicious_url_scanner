from flask import Flask,render_template,url_for,request
import pandas as pd 

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import joblib

# load the model from disk
filename = 'model/model.pkl'
clf = joblib.load(open(filename, 'rb'))
cv  = joblib.load(open('model/tfidfvect.pkl','rb'))
app = Flask(__name__)

def trim(url):
    return re.match(r'(?:\w*://)?(?:.*\.)?([a-zA-Z-1-9]*\.[a-zA-Z]{1,}).*', url).groups()[0]


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():


	if request.method == 'POST':
		message = request.form['url']
		data = [message]
		vect = cv.transform([trim(data)])
		prediction = clf.predict(vect)

		
	return render_template('result.html',prediction = prediction)



if __name__ == '__main__':
	app.run()