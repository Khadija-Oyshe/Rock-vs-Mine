import pickle
from flask import Flask, request, jsonify,app,render_template,url_for
import numpy as np
import pandas as pd

app = Flask(__name__)
##load the model
model = pickle.load(open('classmodel.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')
@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['new_data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data = np.array(list(data.values())).reshape(1,-1)
    output = model.predict(new_data)
    print(output)
    return jsonify(output[0])

if __name__ == "__main__":
    app.run(debug=True)