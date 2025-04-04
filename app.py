from flask import Flask, request, render_template, redirect
import joblib
import pandas as pd
import os
import sqlite3
from datetime import datetime

app = Flask(__name__)

# Load model and encoders
model = joblib.load("model.pkl")
encoders = joblib.load("encoders.pkl")

# Initialize DB if not exists
def init_db():
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            actor_name TEXT,
            predicted_rating REAL,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    title = request.form["movie_title"]
    actor = request.form["actor_name"]

    # Case-insensitive encoder transform
    def safe_transform(field, value):
        value = value.lower().strip()
        encoder = encoders[field]
        classes = [cls.lower() for cls in encoder.classes_]
        if value in classes:
            return encoder.transform([encoder.classes_[classes.index(value)]])[0]
        return 0  # Default if not found

    input_data = {
        "movie_title": title,
        "actor_1_name": safe_transform("actor_1_name", actor)
    }

    df = pd.DataFrame([input_data])
    prediction = model.predict(df)[0]

    # Save to database
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO predictions (title, actor_name, predicted_rating, timestamp)
        VALUES (?, ?, ?, ?)
    """, (title, actor, prediction, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

    return render_template("index.html", prediction=f"{prediction:.2f}")

@app.route("/dashboard")
def dashboard():
    conn = sqlite3.connect("predictions.db")
    df = pd.read_sql_query("SELECT * FROM predictions ORDER BY timestamp DESC", conn)
    conn.close()
    return render_template("dashboard.html", data=df.to_dict(orient="records"))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
