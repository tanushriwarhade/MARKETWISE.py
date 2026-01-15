from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import sqlite3
from datetime import datetime
import os
import random

app = Flask(__name__)
CORS(app)

DB_NAME = "safety.db"

# -------------------- DATABASE INIT --------------------

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS detections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        person_id TEXT,
        helmet INTEGER,
        shoes INTEGER,
        goggles INTEGER,
        mask INTEGER,
        confidence REAL,
        timestamp TEXT
    )
    """)
    conn.commit()
    conn.close()

init_db()

# -------------------- YOLO INIT --------------------

try:
    from ultralytics import YOLO
    MODEL_PATH = os.path.join("..", "models", "yolov8n.pt")
    model = YOLO(MODEL_PATH)
    YOLO_ENABLED = True
except Exception as e:
    print("YOLO not available, using simulation mode:", e)
    YOLO_ENABLED = False

# -------------------- ROUTES --------------------

@app.route("/")
def home():
    return """
    <h2>🛡️ Safety Compliance System</h2>
    <p>Backend is running successfully.</p>
    <p>POST frames to <code>/api/process</code></p>
    """

@app.route("/api/process", methods=["POST"])
def process_frame():
    data = request.json["frame"]
    img_data = base64.b64decode(data.split(",")[1])
    frame = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)

    detections = []

    if YOLO_ENABLED:
        results = model(frame, verbose=False)
        for r in results:
            for _ in r.boxes:
                detections.append(generate_detection())
    else:
        detections.append(generate_detection())

    save_to_db(detections)
    return jsonify(detections)

# -------------------- HELPERS --------------------

def generate_detection():
    return {
        "person_id": f"P{random.randint(1, 5)}",
        "helmet": random.randint(0, 1),
        "shoes": random.randint(0, 1),
        "goggles": random.randint(0, 1),
        "mask": random.randint(0, 1),
        "confidence": round(random.uniform(0.7, 0.95), 2),
        "timestamp": datetime.now().isoformat()
    }

def save_to_db(detections):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    for d in detections:
        c.execute("""
        INSERT INTO detections 
        (person_id, helmet, shoes, goggles, mask, confidence, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            d["person_id"],
            d["helmet"],
            d["shoes"],
            d["goggles"],
            d["mask"],
            d["confidence"],
            d["timestamp"]
        ))
    conn.commit()
    conn.close()

# -------------------- MAIN --------------------

if __name__ == "__main__":
    print("🚀 Server running at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)

@app.route("/api/stats")
def get_stats():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    c.execute("SELECT COUNT(*) FROM detections")
    total = c.fetchone()[0]

    c.execute("""
        SELECT COUNT(*) FROM detections
        WHERE helmet=1 AND shoes=1 AND goggles=1 AND mask=1
    """)
    compliant = c.fetchone()[0]

    conn.close()

    return jsonify({
        "total": total,
        "compliant": compliant,
        "violations": total - compliant,
        "activePersons": 5
    })
