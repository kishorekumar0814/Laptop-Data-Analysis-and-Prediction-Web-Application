from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
from model import load_data, trend_analysis

app = Flask(__name__)

# Load the dataset once at the start
data = pd.read_csv('data.csv')

# Extracting brand names from the 'Name' column
data['Brand'] = data['Name'].apply(lambda x: x.split()[0])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    rating = float(request.form['rating'])
    ratings = int(request.form['ratings'])
    model = joblib.load('price_model.pkl')
    features = np.array([[rating, ratings]])
    prediction = model.predict(features)
    return render_template('result.html', result=f'Predicted Price: ₹{prediction[0]:.2f}')

@app.route('/predict_rating', methods=['POST'])
def predict_rating():
    price = float(request.form['price'])
    ratings = int(request.form['ratings'])
    model = joblib.load('rating_model.pkl')
    features = np.array([[price, ratings]])
    prediction = model.predict(features)
    return render_template('result.html', result=f'Predicted Customer Rating: {prediction[0]:.2f}')


@app.route('/search', methods=['GET'])
def search():
    brand = request.args.get('brand')
    filtered_data = data[data['Brand'].str.contains(brand, case=False, na=False)]
    laptops = filtered_data.to_dict(orient='records')
    return render_template('search_results.html', brand=brand, laptops=laptops)

# Convert 'Date First Available' to datetime
try:
    data['Date First Available'] = pd.to_datetime(data['Date First Available'], errors='coerce')
    print(data['Date First Available'].head())  # Print to verify conversion
except Exception as e:
    print(f"Error converting 'Date First Available' to datetime: {e}")
          
@app.route('/analysis')
def analysis():
    try:
        data['Year'] = data['Date First Available'].dt.year
        year_counts = data['Year'].value_counts().sort_index().to_dict()

        # Calculate price trends (average price per year)
        trends = data.groupby('Year')['Price'].mean().sort_index().to_dict()

        return render_template('analysis.html', year_counts=year_counts, trends=trends)
    except AttributeError as e:
        return f"Error in analysis: {e}"

if __name__ == '__main__':
    app.run(debug=True)
