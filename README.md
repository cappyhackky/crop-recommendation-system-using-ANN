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
    git clone https://github.com/cappyhackky/crop-recommendation-system-using-ANN.git
    ```
2. **Navigate to the project directory:**
    ```bash
    cd crop-recommendation-system-using-ANN
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
