from flask import Flask, request, render_template, jsonify
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'models'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(MODEL_FOLDER):
    os.makedirs(MODEL_FOLDER)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        return train_and_evaluate(file_path)

def train_and_evaluate(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)

    # Check for missing values
    if df.isnull().sum().any():
        return "Dataset contains missing values. Please clean the data and try again."

    # Visualize Price vs Units Sold
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x="Price", y="Units_Sold", hue="Product_Category", data=df)
    plt.title("Price vs Units Sold")
    plt.savefig(os.path.join(UPLOAD_FOLDER, 'visualization.png'))

    # Feature Selection
    X = df[["Price", "Competitor_Price", "Customer_Rating", "Demand_Elasticity"]]
    y = df["Units_Sold"]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest Regressor
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Predictions and Evaluation
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    # Save the model
    model_path = os.path.join(MODEL_FOLDER, "pricing_model.pkl")
    joblib.dump(model, model_path)

    return jsonify({
        "message": "Training completed successfully!",
        "mean_squared_error": mse,
        "model_path": model_path,
        "visualization": os.path.join(UPLOAD_FOLDER, 'visualization.png')
    })

@app.route('/download_model', methods=['GET'])
def download_model():
    model_path = os.path.join(MODEL_FOLDER, "pricing_model.pkl")
    if os.path.exists(model_path):
        return send_file(model_path, as_attachment=True)
    else:
        return "Model file not found."

if __name__ == '__main__':
    app.run(debug=True)
