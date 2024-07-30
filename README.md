# Laptop Data Analysis and Prediction Web Application

## Overview
This project is a web application designed to analyze and predict laptop prices and customer ratings based on a dataset collected from Amazon. Users can search for laptops by brand, predict prices and customer ratings, and analyze trends over time. The application is built using the Flask framework for web development and the Pandas library for data processing.

## Features
- **Search by Brand:** Enter a brand name to get detailed information about available laptops in the dataset.
- **Price Prediction:** Predict the price of a laptop based on customer rating and the number of ratings.
- **Customer Rating Prediction:** Predict the customer rating of a laptop based on its price and the number of ratings.
- **Trend Analysis:** Analyze price trends over time for different laptop brands.

## Technologies Used
- **Flask:** A lightweight Python web framework for building the web application.
- **Pandas:** A data manipulation library in Python for processing and analyzing data.
- **scikit-learn:** A machine learning library in Python for predictive analysis.
- **Jinja2:** A templating engine for rendering HTML templates.
- **Joblib:** A library for serializing and deserializing Python objects.

## Requirements
- Python 3.7 or higher
- Flask
- Pandas
- scikit-learn
- Joblib

## Installation
### Clone the Repository
    git clone https://github.com/kishorekumar0814/laptop-data-analysis.git
    cd laptop-data-analysis

## Set Up a Virtual Environment (Recommended)
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`

## Install Required Packages
    pip install -r requirements.txt

## Configuration
### Create a .env File (Optional, for custom configurations)
#### Create a .env file in the root directory of the project if you need to set environment variables. For example:
    FLASK_ENV=development

## Running the Application
### Start the Flask Server
    python app.py
By default, the Flask server will run on http://127.0.0.1:5000/.

## Access the Application
### Open your web browser and navigate to http://127.0.0.1:5000/. You will see the main interface for searching and analyzing laptops.

## Upload a Dataset
### Use the provided forms to input data for predicting prices and customer ratings. You can also search for laptops by brand and analyze trends.

## Project Structure
    laptop-data-analysis/
    │
    ├── app.py                  # Main application file
    ├── model.py                # Model training and prediction logic
    ├── templates/
    │   ├── analysis.html       # HTML template for trend analysis
    │   ├── index.html          # HTML template for home page
    │   └── result.html         # HTML template for displaying search results
    ├── static/
    │   └── styles.css          # CSS for styling (if any)
    ├── price_model.pkl         # Trained model for price prediction
    ├── rating_model.pkl        # Trained model for customer rating prediction
    ├── requirements.txt        # List of required Python packages
    └── README.md               # This file

## Testing
### Run Unit Tests (if applicable)
#### If you have unit tests in your project, you can run them using:
    pytest

## Troubleshooting
Missing Dependencies: Ensure all required packages are installed. Run pip install -r requirements.txt to install them.
File Upload Issues: Verify that the file format is supported (CSV or XLSX) and that the file is not corrupted.

## Contributing
Feel free to contribute to the project by submitting issues or pull requests. For detailed contribution guidelines, please refer to **CONTRIBUTING.md**.

## License
This project is licensed under the **MIT License**. See the **LICENSE** file for more details.

## Contact
Email: [Email](mailto:kishorekumar1409@gmail.com)

LinkedIn: [LinkedIn](https://www.linkedin.com/in/kishorekumar1409/)

## Project Report
For more detailed information about the project, refer to the project report: [Project_Report.docx](https://github.com/user-attachments/files/16403757/Project_Report.docx)

