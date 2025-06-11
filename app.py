from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
model = pickle.load(open("bike_rental_model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Feature name with expected valid ranges
        valid_ranges = {
            "season": (1, 4),
            "yr": (0, 1),
            "mnth": (1, 12),
            "holiday": (0, 1),
            "weekday": (0, 6),
            "workingday": (0, 1),
            "weathersit": (1, 4),
            "temp": (0, 1),
            "atemp": (0, 1),
            "hum": (0, 1),
            "windspeed": (0, 1),
            "casual": (0, float("inf")),  # Must be non-negative
            "registered": (0, float("inf"))  # Must be non-negative
        }

        features = []
        for feature, (min_val, max_val) in valid_ranges.items():
            value = float(request.form.get(feature, 0))

            # Check if value is within valid range
            if not (min_val <= value <= max_val):
                return render_template('index.html', prediction_text=f"Error: {feature} must be between {min_val} and {max_val}")

            features.append(value)

        # Make prediction
        prediction = model.predict([features])[0]

        return render_template('index.html', prediction_text=f"Predicted Bike Rentals: {prediction:.2f}")

    except Exception as e:
        return render_template('index.html', prediction_text="Prediction Failed!")

if __name__ == "__main__":
    app.run(debug=True)
