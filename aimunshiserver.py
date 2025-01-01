from flask import Flask, request, jsonify
import joblib
from pyngrok import ngrok
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load your trained model and scaler
try:
    model = joblib.load('/content/AiMunshi.joblib')
    scaler = joblib.load('/content/Scalar.pkl')
    logging.info("Model and scaler loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model or scaler: {str(e)}")
    raise

# Flask app initialization
app = Flask(__name__)

# Define the item name to code mapping
item_code_mapping = {
    'apples': 0,
    'bananas': 1,
    'beef': 2,
    'chicken': 3,
    'cooking oil': 4,
    'eggs': 5,
    'ghee': 6,
    'lentils': 7,
    'milk': 8,
    'mutton': 9,
    'onions': 10,
    'petrol': 11,
    'potatoes': 12,
    'rice': 13,
    'shampoo': 14,
    'soap': 15,
    'sugar': 16,
    'tea': 17,
    'tomatoes': 18,
    'wheat flour': 19,
}

# Print all items in the console
logging.info("Available items and their codes:")
for item, code in item_code_mapping.items():
    logging.info(f"{item}: {code}")

@app.route('/')
def index():
    return jsonify({"message": "Welcome to the prediction API"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract Dialogflow parameters from the request
        data = request.json
        if not data:
            return jsonify({"fulfillmentText": "Invalid request: No JSON data provided."})

        query_result = data.get('queryResult', {})
        parameters = query_result.get('parameters', {})

        item_name = parameters.get('items')
        date = parameters.get('date-time')

        if not item_name or not date:
            return jsonify({"fulfillmentText": "Invalid request: Missing 'items' or 'date-time' parameters."})

        # Map the item name to its code
        item_code = item_code_mapping.get(item_name.lower(), None)
        if item_code is None:
            return jsonify({"fulfillmentText": f"Sorry, we do not have data for '{item_name}'. Please try another item."})

        # Parse the date and extract month, day, and year
        try:
            date_obj = datetime.fromisoformat(date)
        except ValueError:
            return jsonify({"fulfillmentText": "Invalid date format. Please provide the date in ISO format (YYYY-MM-DD)."})

        month = date_obj.month
        day = date_obj.day
        year = date_obj.year

        # Prepare input for the model
        model_input = [[month, day, year, item_code]]

        # Scale the input features
        scaled_input = scaler.transform(model_input)

        # Predict the price
        predicted_price = model.predict(scaled_input)

        # Format the response
        formatted_date = date_obj.strftime('%d/%m/%Y')
        rounded_price = round(predicted_price[0], 1)

        return jsonify({
            "fulfillmentText": f"The predicted price of {item_name} on {formatted_date} is PKR {rounded_price}."
        })

    except KeyError as ke:
        logging.error(f"KeyError: {str(ke)}")
        return jsonify({"fulfillmentText": "An error occurred: Missing data fields."})

    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        return jsonify({"fulfillmentText": f"An error occurred while processing your request: {str(e)}"})

if __name__ == '__main__':
    try:
        # Start ngrok tunnel to expose Flask app
        public_url = ngrok.connect(8000)
        logging.info(f"ngrok tunnel created: {public_url}")
        print(f" * ngrok tunnel \"{public_url}\" -> \"http://127.0.0.1:8000\"")
    except Exception as e:
        logging.error(f"Error starting ngrok tunnel: {str(e)}")
        raise

    app.run(host='127.0.0.1', port=8000)
