from flask import Flask, request, render_template, redirect
import joblib
import pandas as pd
import sqlite3
import os

app = Flask(__name__)

# Load model and encoders
model = joblib.load("model.pkl")
encoders = joblib.load("encoders.pkl")

# Initialize SQLite DB
DB_FILE = "predictions.db"
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT,
                    director TEXT,
                    actor TEXT,
                    genre TEXT,
                    year INTEGER,
                    predicted_rating REAL
                )''')
    conn.commit()
    conn.close()
init_db()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get and clean inputs
    title = request.form["movie_title"].strip().lower()
    director = request.form["director_name"].strip().lower()
    actor = request.form["actor_name"].strip().lower()
    genre = request.form["genre"].strip().lower()
    year = int(request.form["release_year"])

    # Encode categorical fields
    def safe_encode(encoder, value):
        value = value.lower()
        return encoder.transform([value])[0] if value in encoder.classes_ else 0

    input_data = {
        "movie_title": title,
        "director_name": safe_encode(encoders["director_name"], director),
        "actor_1_name": safe_encode(encoders["actor_1_name"], actor),
        "genres": safe_encode(encoders["genres"], genre),
        "title_year": year
    }

    df = pd.DataFrame([input_data])
    predicted_rating = model.predict(df)[0]

    # Save to DB
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO predictions (title, director, actor, genre, year, predicted_rating) VALUES (?, ?, ?, ?, ?, ?)",
              (title, director, actor, genre, year, predicted_rating))
    conn.commit()
    conn.close()

    return render_template("index.html", prediction=f"Predicted Rating: {predicted_rating:.2f}")

@app.route("/dashboard")
def dashboard():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT * FROM predictions ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()
    return render_template("dashboard.html", rows=rows)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
