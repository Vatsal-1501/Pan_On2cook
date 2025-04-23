# On2Cook AI Server

This repository contains the On2Cook AI server, which uses machine learning models to detect the status of a cooking pan and predict its temperature. The server is implemented using Flask and is designed to be hosted on Render for cloud-based inference. An ESP32 device running FreeRTOS is used to collect sensor data and send it to the server for real-time analysis.

## Features
- **Pan Status Detection:** Determines if a pan is empty or not.
- **Temperature Prediction:** Predicts the pan's temperature based on sensor data.
- **Flask API:** A lightweight server to handle HTTP requests.
- **Render Deployment:** Cloud-hosted inference using Render.
- **ESP32 Integration:** Sends sensor data to the server and processes responses using FreeRTOS tasks.

---

## 1. Model Details

### Machine Learning Models
- **`pan_classifier.pkl`**: A classification model that determines if the pan is empty or not.
- **`temperature_predictor.pkl`**: A regression model that predicts the pan temperature.
- **`scaler.pkl`**: A scaler to normalize input features before passing them to the models.

### Input Features
The models use the following sensor data:
1. **PAN_Inside** (Oil Temperature)
2. **PAN_Outside** (Coil Temperature)
3. **Glass_Temp** (Glass Surface Temperature)
4. **Ind_Current** (Induction Current)
5. **Mag_Current** (Magnetron Current)

### API Response
- **`pan_status`**: "Not Empty" or "Empty"
- **`temperature`**: Predicted temperature value

---

## 2. Hosting on Render

### Steps to Deploy
1. **Create a Render Account** at [https://render.com](https://render.com).
2. **New Web Service**: Click on "New Web Service" and connect your GitHub repository.
3. **Select Python Environment**: Choose a Python runtime, e.g., `python-3.9`.
4. **Install Dependencies**:
   ```sh
   pip install -r requirements.txt
   ```
5. **Set up Environment Variables**:
   - `FLASK_ENV=production`
   - `PORT=5000`
6. **Deploy**: Click on "Deploy" and wait for Render to build and launch the service.
7. **Test API**:
   ```sh
   curl -X POST "https://your-render-url.onrender.com/detect" -H "Content-Type: application/json" -d '{"features": [50.5, 48.3, 35.0, 1.2, 0.0]}'
   ```

---

## 3. ESP32 Integration

The ESP32 collects sensor data and sends it to the AI server for inference.

### Code Overview (ESP32)
- Connects to Wi-Fi.
- Reads sensor data.
- Formats and sends JSON data to the server.
- Parses and prints the AI response.
- Runs inside a FreeRTOS task for real-time execution.

### Dependencies
- **WiFi.h**: For network connectivity.
- **HTTPClient.h**: To send HTTP requests.
- **ArduinoJson.h**: To handle JSON serialization.

### FreeRTOS Task for HTTP Requests
```cpp
xTaskCreatePinnedToCore(
    httpTask, "HTTP Task", 10000, NULL, 1, &httpTaskHandle, 0);
```
This ensures non-blocking execution while sending data.

### API Request from ESP32
```cpp
float features[5] = {oil_temp, coil_temp, Indcontroller_GlassTemp(), Indcontroller_IndCurr(), Mag_cur};
```

### Example API Response
```json
{
  "pan_status": "Not Empty",
  "temperature": 180.5
}
```

### How to Flash ESP32 Code
1. **Install ESP32 Board in Arduino IDE / PlatformIO**.
2. **Replace Wi-Fi Credentials** in `new.cpp`:
   ```cpp
   const char* ssid = "YOUR_WIFI_SSID";
   const char* password = "YOUR_WIFI_PASSWORD";
   ```
3. **Flash Code** to ESP32:
   ```sh
   pio run -t upload
   ```
4. **Monitor Serial Output**:
   ```sh
   pio device monitor
   ```

---

## 4. Running Locally

### Setup
1. **Install Dependencies**
   ```sh
   pip install -r requirements.txt
   ```
2. **Run Flask Server**
   ```sh
   python app.py
   ```
3. **Test with CURL**
   ```sh
   curl -X POST "http://127.0.0.1:5000/detect" -H "Content-Type: application/json" -d '{"features": [50.5, 48.3, 35.0, 1.2, 0.0]}'
   ```

---

## 5. Future Enhancements
- **Optimize Model Inference**: Deploy a TensorFlow Lite model for faster execution.
- **Secure API**: Implement API authentication.
- **ESP32 OTA Updates**: Enable over-the-air firmware updates.

---

## 6. Repository Structure
```
/
|-- app.py            # Flask server
|-- requirements.txt  # Dependencies
|-- models/
|   |-- pan_classifier.pkl
|   |-- temperature_predictor.pkl
|   |-- scaler.pkl
|-- esp32/
|   |-- new.cpp       # ESP32 FreeRTOS HTTP code
```

---

## 7. License
This project is open-source under the MIT License.

