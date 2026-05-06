from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

model = pickle.load(open("model/heart_disease_model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["input"]

    # ✅ EXACT columns (NO "Sex")
    columns = [
        "Age",
        "Chest pain type",
        "BP",
        "Cholesterol",
        "FBS over 120",
        "EKG results",
        "Max HR",
        "Exercise angina",
        "ST depression",
        "Slope of ST",
        "Number of vessels fluro",
        "Thallium"
    ]

    df = pd.DataFrame([data], columns=columns)

    input_scaled = scaler.transform(df)

    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    return jsonify({
        "result": "High Risk" if prediction == 1 else "Low Risk",
        "probability": round(prob * 100, 2)
    })

if __name__ == "__main__":
    app.run(debug=True)