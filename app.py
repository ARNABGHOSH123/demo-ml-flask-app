from flask import Flask, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)
# Load the model
with open('model.pkl', 'rb') as f:
  model = pickle.load(f)

version = os.environ.get("VERSION", "dev")

@app.route('/')
def home():
    return jsonify({'message': f'Sales Prediction API is running with all task 1, task 2 and task 3 completed and is ready for canary deployment, version : {version}'}) 


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({'predicted_sales': prediction[0]})


if __name__ == '__main__':
  app.run(host='0.0.0.0', port=8080)
