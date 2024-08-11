import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from flask import Flask, request, render_template
import joblib
from datetime import datetime, timedelta

app = Flask(__name__)

class SalesForecaster:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.scaler = StandardScaler()
        self.kmeans = None
        self.rf_model = None
        self.feature_names = ['Order Quantity', 'Discount %', 'Shipping Cost', 'DayOfWeek', 'Month', 'Year']

    def clean_percentage(self, x):
        if isinstance(x, str):
            try:
                return float(x.rstrip('%')) / 100
            except ValueError:
                return np.nan
        return x

    def clean_currency(self, x):
        if isinstance(x, str):
            try:
                return float(x.replace('$', '').replace(',', ''))
            except ValueError:
                return np.nan
        return x

    def preprocess(self):
        # Convert 'Order Date' to datetime, trying multiple formats
        self.df['Order Date'] = pd.to_datetime(self.df['Order Date'], format='mixed', dayfirst=True)
        
        # Extract time-based features
        self.df['DayOfWeek'] = self.df['Order Date'].dt.dayofweek
        self.df['Month'] = self.df['Order Date'].dt.month
        self.df['Year'] = self.df['Order Date'].dt.year

        # Select features for clustering and forecasting
        X = self.df[self.feature_names].copy()
        
        # Clean the 'Discount %' and 'Shipping Cost' columns
        X['Discount %'] = X['Discount %'].apply(self.clean_percentage)
        X['Shipping Cost'] = X['Shipping Cost'].apply(self.clean_currency)
        
        # Convert to numeric and handle any remaining non-numeric values
        for col in self.feature_names:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        
        # Clean the 'Total' column
        self.df['Total'] = self.df['Total'].apply(self.clean_currency)
        
        # Drop any rows with NaN values
        X = X.dropna()
        self.df = self.df.loc[X.index]  # Align the main dataframe with X
        
        if X.empty:
            raise ValueError("After preprocessing, no valid data remains. Please check your input data.")
        
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        return X, X_scaled

    def train_model(self, n_clusters=5):
        X, X_scaled = self.preprocess()
        
        # Perform K-means clustering
        self.kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        cluster_labels = self.kmeans.fit_predict(X_scaled)
        
        # Add cluster labels to the feature set
        X['Cluster'] = cluster_labels
        
        # Prepare data for the Random Forest model
        if 'Total' not in self.df.columns:
            raise ValueError("'Total' column is missing from the CSV file. Please ensure it exists.")
        y = self.df['Total']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train Random Forest model
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.rf_model.fit(X_train, y_train)
        
        # Print model performance
        train_score = self.rf_model.score(X_train, y_train)
        test_score = self.rf_model.score(X_test, y_test)
        print(f"Train R2 Score: {train_score}")
        print(f"Test R2 Score: {test_score}")

    def predict(self, input_data):
        # Ensure the scaler is fitted
        if not hasattr(self.scaler, 'mean_'):
            raise ValueError("The model has not been trained yet. Please call train_model() first.")
        
        # Convert input_data to DataFrame
        input_df = pd.DataFrame([input_data], columns=self.feature_names)
        
        # Scale the input data
        input_scaled = self.scaler.transform(input_df)
        
        # Find the nearest cluster
        cluster = self.kmeans.predict(input_scaled)[0]
        
        # Add cluster to input data
        input_df['Cluster'] = cluster
        
        # Make prediction
        prediction = self.rf_model.predict(input_df)[0]
        
        return prediction

# Initialize and train the model
sales_forecaster = SalesForecaster('sales_data.csv')
try:
    sales_forecaster.train_model()
except Exception as e:
    print(f"An error occurred during model training: {str(e)}")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            order_date = datetime.strptime(request.form['order_date'], '%Y-%m-%d')
            input_data = [
                float(request.form['order_quantity']),
                float(request.form['discount_percent']) / 100,
                float(request.form['shipping_cost']),
                order_date.weekday(),
                order_date.month,
                order_date.year
            ]

            prediction = sales_forecaster.predict(input_data)
            return render_template('result.html', prediction=prediction)
        except Exception as e:
            return render_template('error.html', error=str(e))

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
