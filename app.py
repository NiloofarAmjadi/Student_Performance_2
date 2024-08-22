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
    # Collect data from form submission
    math_score = request.form['math']
    reading_score = request.form['reading']
    writing_score = request.form['writing']

    # Create a DataFrame
    data = {
        'math score': [math_score],
        'reading score': [reading_score],
        'writing score': [writing_score]
    }
    df = pd.DataFrame(data)

    # Make prediction
    prediction = model.predict(df)

    # Return the result
    return render_template('index.html', predict=prediction[0])


# Run the app
if __name__ == "__main__":
    app.run(debug=True)