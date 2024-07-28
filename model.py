import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import numpy as np
import warnings

# Suppress FutureWarning messages
warnings.simplefilter(action='ignore', category=FutureWarning)


def load_data():
    data = pd.read_csv('data.csv')

    data['Date First Available'] = pd.to_datetime(data['Date First Available'], errors='coerce')

    data['Price'].fillna(data['Price'].mean(), inplace=True)
    data['Customer Rating'].fillna(data['Customer Rating'].mean(), inplace=True)
    data['Number of Ratings'].fillna(data['Number of Ratings'].mean(), inplace=True)
    
    return data

data = load_data()

def train_price_model(data):
    features = ['Customer Rating', 'Number of Ratings']
    X = data[features]
    y = data['Price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f'Price Prediction Model - MSE: {mse}, R2: {r2}')
    
    joblib.dump(model, 'price_model.pkl')

def train_rating_model(data):
    features = ['Price', 'Number of Ratings']
    X = data[features]
    y = data['Customer Rating']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f'Customer Rating Model - MSE: {mse}, R2: {r2}')
    
    joblib.dump(model, 'rating_model.pkl')

def trend_analysis(data):
    data['Year'] = data['Date First Available'].dt.year
    yearly_avg_price = data.groupby('Year')['Price'].mean().dropna()
    return yearly_avg_price.to_dict()

train_price_model(data)
train_rating_model(data)