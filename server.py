from flask import Flask, jsonify, request
import pandas as pd
import joblib

app = Flask(__name__)

@app.route("/predict", methods=['POST'])
def make_prediction():
	json = request.get_json()
	model = joblib.load('model/model.pkl')
	# df = pd.DataFrame(json, index=[0])
	y_predict = model.predict({json['prediction']})
	result = {"Disaster predicted?" : str(y_predict[0])}
	return result


if __name__ == "__main__":
	app.run(host='0.0.0.0')