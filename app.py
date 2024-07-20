from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import json
import tensorflow as tf
import numpy as np

app = Flask(__name__)
CORS(app)

loaded_model = tf.keras.models.load_model('crop_recom_ann.h5')

crop_dictionary = {
    20: 'Rice',
    11: 'Maize',
    3: 'Chickpea',
    9: 'Kidneybeans',
    18: 'Pigeonpeas',
    13: 'Mothbeans',
    14: 'Mungbean',
    2: 'Blackgram',
    10: 'Lentil',
    19: 'Pomegranate',
    1: 'Banana',
    12: 'Mango',
    7: 'Grapes',
    21: 'Watermelon',
    15: 'Muskmelon',
    0: 'Apple',
    16: 'Orange',
    17: 'Papaya',
    4: 'Coconut',
    6: 'Cotton',
    8: 'Jute',
    5: 'Coffee'
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit_data', methods=['POST'])
def getone():
    try:
        data = json.loads(request.data)
        values = list(data.values())
        print(data)
        print(values)

        for_pred = np.array(values, dtype=float).reshape(1, -1)
        print(for_pred)

        result = loaded_model.predict(for_pred)
        print(result)

        predicted_class_index = np.argmax(result, axis=-1)[0]

        predicted_crop = crop_dictionary.get(predicted_class_index, "Unknown")
        # print("hiii")
        response_data = {
            'predicted_crop': predicted_crop
        }
        print(jsonify(response_data))
        return jsonify(response_data)

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
