from flask import Flask, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)

# Load models
status_model = joblib.load("pan_classifier.pkl")
temp_model = joblib.load("temperature_predictor.pkl")
scaler = joblib.load("scaler.pkl")  # Load the scaler

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "On2Cook AI Server is Running!"})

@app.route("/detect", methods=["POST"])
def detect_pan_status():
    try:
        data = request.json  # Receive JSON sensor readings
        features = np.array(data["features"]).reshape(1, -1)
        
        # Scale input features
        features_scaled = scaler.transform(features)
        
        # Predict pan status
        status = status_model.predict(features)[0]  # 0 (Empty) or 1 (Not Empty)
        
        # Predict temperature
        temperature = temp_model.predict(features)[0]
        
        return jsonify({"pan_status": "Not Empty" if status == 1 else "Empty", "temperature": round(temperature, 2)})
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)


