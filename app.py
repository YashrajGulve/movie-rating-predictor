from flask import Flask, request, render_template
import joblib
import pandas as pd
import os  # ✅ Add this line

app = Flask(__name__)

# Load model and encoders
model = joblib.load("model.pkl")
encoders = joblib.load("encoders.pkl")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    input_data = {
        "director_name": request.form["director_name"],
        "actor_1_name": request.form["actor_1_name"],
        "actor_2_name": request.form["actor_2_name"],
        "actor_3_name": request.form["actor_3_name"],
        "genres": request.form["genres"],
        "duration": float(request.form["duration"]),
        "budget": float(request.form["budget"]),
        "title_year": int(request.form["title_year"]),
    }

    for field in ["director_name", "actor_1_name", "actor_2_name", "actor_3_name", "genres"]:
        encoder = encoders[field]
        value = input_data[field]
        input_data[field] = encoder.transform([value])[0] if value in encoder.classes_ else 0

    df = pd.DataFrame([input_data])
    prediction = model.predict(df)[0]
    return render_template("index.html", prediction=f"{prediction:.2f}")

    import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
