from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

cover_map = {
    1: "Spruce/Fir",
    2: "Lodgepole Pine",
    3: "Ponderosa Pine",
    4: "Cottonwood/Willow",
    5: "Aspen",
    6: "Douglas-fir",
    7: "Krummholz"
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    features = [float(x) for x in request.form.values()]
    prediction = model.predict([features])[0]
    return render_template(
        "index.html",
        prediction_text=f"Predicted Forest Cover Type: {cover_map[prediction]}"
    )

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.json["features"]
    prediction = model.predict([data])[0]
    return jsonify({"cover_type": int(prediction)})

if __name__ == "__main__":
    app.run()
