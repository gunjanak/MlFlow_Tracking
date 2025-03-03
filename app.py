from flask import Flask, request, render_template, jsonify
import requests
from PIL import Image
import numpy as np
from sklearn.preprocessing import StandardScaler
from joblib import load

# Load the scaler
scaler = load("scaler.joblib")

app = Flask(__name__)

# MLflow model endpoint
MLFLOW_MODEL_URL = "http://localhost:1234/invocations"



# Preprocess the image (resize, normalize, and apply scalar transformation)
def preprocess_image(image):
    # Resize to 8x8 (digits dataset image size)
    image = image.resize((8, 8))
    # Convert to grayscale
    image = image.convert("L")
    # Convert to numpy array
    image_array = np.array(image)
    # Normalize pixel values to [0, 16] (as in the digits dataset)
    image_array = (image_array / 255.0) * 16
    # Flatten the array to match the model's input shape
    image_array = image_array.flatten()
    # Apply scalar transformation (if applicable)
    image_array = scaler.transform([image_array])  # Reshape to 2D array for scaler
    # Now you can use the scaler to transform input data
    transformed_data = scaler.transform(image_array)
    print(f"Image Array: {image_array}")
    print(f"Transformed data:{transformed_data}")
    return image_array.tolist()[0]  # Return the transformed 1D array

@app.route("/", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":
        # Check if a file was uploaded
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        try:
            # Read the image file
            image = Image.open(file.stream)
            # Preprocess the image
            processed_image = preprocess_image(image)
            # Prepare the input for the MLflow model
            data = {
                "inputs": [processed_image]
            }
            # Send the request to the MLflow model
            response = requests.post(MLFLOW_MODEL_URL, json=data)
            if response.status_code == 200:
                # Get the prediction
                prediction = response.json().get("predictions", [None])[0]
                print(f"prediction: {prediction}")
                return jsonify({"prediction": prediction})
            else:
                return jsonify({"error": "Failed to get prediction from model", "details": response.text}), 500
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # Render the upload form for GET requests
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)