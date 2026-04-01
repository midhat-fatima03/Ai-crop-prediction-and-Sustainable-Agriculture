from flask import Flask, render_template, request
import pickle
import numpy as np
import os
import pandas as pd

# -------------------------------
# 1️⃣ Initialize Flask
# -------------------------------
app = Flask(__name__)

# -------------------------------
# 2️⃣ Load trained model
# -------------------------------
model_path = os.path.join(os.getcwd(), "model.pkl")

with open(model_path, "rb") as f:
    model = pickle.load(f)

# Graph file names (NOT full path)
accuracy_graph = "accuracy_graph.png"
feature_graph = "feature_importance.png"

# -------------------------------
# 3️⃣ Home route
# -------------------------------
@app.route('/')
def home():
    return render_template("index.html")

# -------------------------------
# 4️⃣ Prediction route
# -------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # Input validation
        if any(val < 0 for val in [N, P, K, temperature, humidity, ph, rainfall]):
            return "All inputs must be non-negative numbers!"

        # Prepare input (FIXED INDENTATION)
        input_features = pd.DataFrame(
            [[N, P, K, temperature, humidity, ph, rainfall]],
            columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        )

        # Prediction
        predicted_crop = model.predict(input_features)[0]

        # Explanation
        explanation = {
            "N": N,
            "P": P,
            "K": K,
            "Temperature": f"{temperature} °C",
            "Humidity": f"{humidity} %",
            "pH": ph,
            "Rainfall": f"{rainfall} mm",
            "Reason": f"Based on these conditions, the model predicts {predicted_crop} as the most suitable crop."
        }

        return render_template(
            "result.html",
            crop=predicted_crop,
            explanation=explanation,
            accuracy_graph=accuracy_graph,
            feature_graph=feature_graph
        )

    except ValueError:
        return "Please enter valid numeric values for all fields."

# -------------------------------
# 5️⃣ Run app
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)




