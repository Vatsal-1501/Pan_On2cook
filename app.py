from flask import Flask, request, jsonify
import numpy as np
import joblib
import pandas as pd

app = Flask(__name__)

# Load models
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "On2Cook AI Server is Running!"})

@app.route("/detect", methods=["POST"])
def detect_pan_status():
    try:
        # Get input data
        data = request.json
        features = np.array([
            data.get("PAN_Inside", 0),
            data.get("PAN_Outside", 0),
            data.get("Glass_Temp", 0)
        ]).reshape(1, -1)
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Predict
        prediction = model.predict(features_scaled)[0]
        
        # Return result
        return jsonify({
            "pan_status": "Empty" if prediction == 1 else "Not Empty",
            "confidence": float(model.predict_proba(features_scaled).max())
        })
    
    except Exception as e:
        return jsonify({"error": str(e)})

if _name_ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)