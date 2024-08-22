from flask import Flask, request, jsonify,render_template
import pandas as pd
import pickle

# Create a Flask app
app = Flask(__name__)
filename = 'Random_Forest_model.pkl'
model = pickle.load(open(filename, 'rb'))    # load the model

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame(data, index=[0])
    prediction = model.predict(df)
    return jsonify({'race/ethnicity': prediction[0]})

# Run the app
if __name__ == "__main__":
    app.run(debug=True)