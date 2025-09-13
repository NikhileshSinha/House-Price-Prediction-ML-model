import pandas as pd
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
data = pd.read_csv('HousingCleanData.csv')

# Load the pipeline model
with open('RidgeModel02.pkl', 'rb') as file:
    pipe = pickle.load(file)

# Ensure the model is correctly loaded
print(f"Loaded model type: {type(pipe)}")

@app.route('/')
def index():
    loc = sorted(data['location'].unique())
    return render_template('indexTwo.html', loc=loc)

@app.route('/predict', methods=['POST'])
def predict():
    location = request.form.get('location')
    bhk = request.form.get('bhk')
    bath = request.form.get('bath')
    sqft = request.form.get('total_sqft')
    
    # Check for empty or missing values and handle them
    if not location or not bhk or not bath or not sqft:
        return "All fields are required!"
    
    try:
        bhk = float(bhk)
        bath = float(bath)
        sqft = float(sqft)
    except ValueError:
        return "Please enter valid numerical values for BHK, Bath, and Total Square Feet."
    
    myloc = data['location'][0]
    print(f"\n\nmyloc:{myloc} and it's type: {type(myloc)}\n\n")
    print(location, bhk, bath, sqft)
    input_data = pd.DataFrame([[location, sqft, bath, bhk]], columns=['location', 'total_sqft', 'bath', 'bhk'])
    
    # Debug: Ensure the input DataFrame is correct
    # print(f"Input DataFrame:\n{input_data}")
    print("\n\n\n\n\n\n\n\n\n********************************")
    print(f"loc: {location} bhk:{bhk} bath:{bath} sqft:{sqft}")
    print("\n")
    print(type(location), type(bhk), type(bath), type(sqft))
    print("********************************\n\n\n\n\n\n\n\n\n")

    # Handle unknown categories for location
    try:
        prediction = pipe.predict(input_data)[0] * 1e5
    except ValueError as e:
        return str(e)
    except AttributeError as e:
        # Debug: Print error details
        print(f"AttributeError: {e}")
        return "An error occurred during prediction. Please check the input values and try again."

    return str(np.round(prediction,2))

if __name__ == '__main__':
    app.run(debug=True, port=5000)
