from flask import Flask, request, render_template
import joblib
import pandas as pd
import csv
import os
from datetime import datetime

app = Flask(__name__)

model = joblib.load("model.pkl")
encoders = joblib.load("encoders.pkl")

def log_prediction(input_data, prediction):
    log_data = input_data.copy()
    log_data["prediction"] = prediction
    log_data["timestamp"] = datetime.now().isoformat()

    with open("prediction_log.csv", mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_data.keys())
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow(log_data)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    input_data = {
        "director_name": request.form["director_name"],
        "actor_1_name": request.form["actor_1_name"],
        "movie_title": request.form["movie_title"],
        "title_year": int(request.form["title_year"]),
        "genres": request.form["genres"]
    }

    # Encode categorical fields
    for field in ["director_name", "actor_1_name", "movie_title", "genres"]:
        encoder = encoders.get(field)
        value = input_data[field]
        input_data[field] = encoder.transform([value])[0] if value in encoder.classes_ else 0

    df = pd.DataFrame([input_data])
    prediction = model.predict(df)[0]

    log_prediction(input_data, prediction)

    return render_template("index.html", prediction=f"{prediction:.2f}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
