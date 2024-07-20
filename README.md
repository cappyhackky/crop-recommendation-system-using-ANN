# Smart Crop Recommendation System

## Overview

This is an advanced Artificial Neural Network (ANN) based system designed to assist farmers and agricultural professionals in selecting the most suitable crops for their fields. By analyzing various environmental and soil parameters, CropSage provides accurate crop recommendations, thereby enhancing agricultural productivity and sustainability.

## Features

- **High-Accuracy Predictions:** Leverages a trained ANN model to suggest the best crops for given conditions.
- **User-Friendly Interface:** Simple and intuitive web application built with Flask.
- **Robust Error Handling:** Ensures reliable performance and smooth operation.
- **Efficient Data Processing:** Fast and accurate predictions thanks to optimized data handling techniques.

## Technologies Used

- **TensorFlow & Keras:** For developing and training the ANN model.
- **Flask:** To create the web application interface.
- **NumPy:** For numerical computations and data manipulation.
- **JSON:** For handling data input and output between the frontend and backend.

## Installation

### Prerequisites

- Python 3.x
- Pip (Python package installer)

### Steps

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/cropsage.git
    ```
2. **Navigate to the project directory:**
    ```bash
    cd cropsage
    ```
3. **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```
4. **Run the Flask application:**
    ```bash
    python app.py
    ```

## Usage

1. Open your web browser and go to `http://127.0.0.1:5000/`.
2. Input the required environmental and soil parameters.
3. Submit the data to receive crop recommendations.

## Project Structure

- `app.py`: The main Flask application file.
- `crop_recom_ann.h5`: The trained ANN model file.
- `templates/index.html`: HTML template for the web interface.
- `static/`: Directory for static files (CSS, JavaScript).

## Example Code

```python
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('crop_recom_ann.h5')

# Example input data
input_data = np.array([[80, 40, 20, 30, 6.5, 200, 85]])

# Predict crop suitability
prediction = model.predict(input_data)
predicted_class_index = np.argmax(prediction, axis=-1)[0]

# Crop dictionary
crop_dictionary = {
    20: 'rice', 11: 'maize', 3: 'chickpea', 9: 'kidneybeans',
    18: 'pigeonpeas', 13: 'mothbeans', 14: 'mungbean', 2: 'blackgram',
    10: 'lentil', 19: 'pomegranate', 1: 'banana', 12: 'mango',
    7: 'grapes', 21: 'watermelon', 15: 'muskmelon', 0: 'apple',
    16: 'orange', 17: 'papaya', 4: 'coconut', 6: 'cotton',
    8: 'jute', 5: 'coffee'
}

# Get the predicted crop
predicted_crop = crop_dictionary.get(predicted_class_index, "Unknown")
print(f"Recommended Crop: {predicted_crop}")
