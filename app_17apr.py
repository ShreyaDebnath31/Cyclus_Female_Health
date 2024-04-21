from flask import Flask, request, jsonify
from flask_restful import Resource, Api
import joblib
import numpy as np
from datetime import datetime, timedelta

app = Flask(__name__)
api = Api(app)

# Load the trained model
with open('model_cyclus_21apr.pkl', 'rb') as file:
    model = joblib.load(file)

class PredictOvulation(Resource):
    def post(self):
        # Get input data from the request
        data = request.get_json()

        # Extract input features
        weight = data['weight']
        height = data['height']
        length_of_cycle = data['length_of_cycle']
        start_date_of_last_period_str = data['start_date_of_last_period']
        end_date_of_last_period_str = data['end_date_of_last_period']
        unusual_bleeding = data['unusual_bleeding']

        if unusual_bleeding == '0':
            UnusualBleeding_0 = 1
            UnusualBleeding_1 = 0
        else:
            UnusualBleeding_0 = 0
            UnusualBleeding_1 = 1

        # Parse start and end dates of the last period
        start_date_of_last_period = datetime.strptime(start_date_of_last_period_str, '%Y-%m-%d').date()
        end_date_of_last_period = datetime.strptime(end_date_of_last_period_str, '%Y-%m-%d').date()
        LengthofMenses = (end_date_of_last_period - start_date_of_last_period).days + 1

        # Calculate additional features based on input data
        length_of_luteal_phase = length_of_cycle - LengthofMenses
        
        # Prepare input features for the model
        features = np.array([[length_of_cycle, length_of_luteal_phase, LengthofMenses, weight, height, UnusualBleeding_0, UnusualBleeding_1]])

        # Make prediction using the model
        estimated_day_of_ovulation = model.predict(features)

        # Ensure that the estimated day of ovulation is within a valid range
        estimated_day_of_ovulation = max(0, min(int(estimated_day_of_ovulation[0]), length_of_cycle - LengthofMenses))

        # Calculate next expected start date of period
        estimated_ovulation_date = start_date_of_last_period + timedelta(days=estimated_day_of_ovulation)
        next_start_date_of_period = estimated_ovulation_date + timedelta(days=LengthofMenses)

        # Prepare response
        response = {
            'estimated_day_of_ovulation': estimated_day_of_ovulation,
            'next_start_date_of_period': next_start_date_of_period.strftime("%Y-%m-%d")
        }

        return jsonify(response)

# Add resource to API
api.add_resource(PredictOvulation, '/predict_ovulation')

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')
