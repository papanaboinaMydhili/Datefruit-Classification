from flask import Flask, request, jsonify,render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import joblib

print(tf.__version__) 
app = Flask(__name__)
print("Starting Flask application...")

# Load the saved model
model = load_model('optimized_reduced_nn_model.h5')
label_encoder = joblib.load('label_encoder.pkl')
scaler = joblib.load('scaler1.pkl')


# Define class names
  # Replace with actual class names
@app.route('/')
def home():
    return render_template('index.html')

def predict_class(input_features):
    """
    Predict the class name based on input features.
    """
   
    input_array = np.array(input_features).reshape(1, -1)
    input_data_scaled = scaler.transform(input_array)
    predictions = model.predict(input_data_scaled)
    confidence = np.max(predictions)
    predicted_index = np.argmax(predictions, axis=1)[0]
    predicted_class = label_encoder.inverse_transform([predicted_index])[0]
    
    return predicted_class


# Route to handle form submission
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get feature values from form
        features = request.json
        feature_values = list(features.values())
        

        # Predict the class
        predicted_class = predict_class(feature_values)
        

        # Return the result to the frontend
        return jsonify({
            'prediction': predicted_class,
            'entered_values': features
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)