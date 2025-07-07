from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load trained model and scaler
with open('models/trained_logreg.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler/trained_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [
            float(request.form['AGE']),
            float(request.form['GENDER']),
            float(request.form['SMOKING']),
            float(request.form['FINGER_DISCOLORATION']),
            float(request.form['MENTAL_STRESS']),
            float(request.form['EXPOSURE_TO_POLLUTION']),
            float(request.form['LONG_TERM_ILLNESS']),
            float(request.form['ENERGY_LEVEL']),
            float(request.form['IMMUNE_WEAKNESS']),
            float(request.form['BREATHING_ISSUE']),
            float(request.form['ALCOHOL_CONSUMPTION']),
            float(request.form['THROAT_DISCOMFORT']),
            float(request.form['OXYGEN_SATURATION']),
            float(request.form['CHEST_TIGHTNESS']),
            float(request.form['FAMILY_HISTORY']),
            float(request.form['SMOKING_FAMILY_HISTORY']),
            float(request.form['STRESS_IMMUNE'])
        ]

        features_array = np.array([features])
        scaled_features = scaler.transform(features_array)

        prediction = model.predict(scaled_features)[0]
        probability = model.predict_proba(scaled_features)[0][1]

        result = "✅ Diagnosed with Lung Cancer (Pulmonary Disease)" if probability >= 0.8 else \
                 "🟡 Likely to be diagnosed with Lung Cancer (borderline)" if probability >= 0.6 else \
                 "⚠️ Uncertain" if 0.4 < probability < 0.6 else "❌ Healthy"

        return render_template('result.html', prediction=result, prob=round(probability * 100, 2))

    except Exception as e:
        return f"Error occurred: {e}"

if __name__ == '__main__':
    app.run(debug=True)