import pickle

import numpy as np
from flask import Flask, render_template, request


# Create flask app
flask_app = Flask(__name__)

# Specify the model to be used
model = pickle.load(open("wine_svm_model.pkl", "rb"))


@flask_app.route("/")
def Home():
    return render_template("index.html")


@flask_app.route("/predict", methods=["POST"])
def predict():
    # Get form values
    fixed_acidity = float(request.form["fixed_acidity"])
    volatile_acidity = float(request.form["volatile_acidity"])
    citric_acid = float(request.form["citric_acid"])
    residual_sugar = float(request.form["residual_sugar"])
    chlorides = float(request.form["chlorides"])
    free_sulfur_dioxide = float(request.form["free_sulfur_dioxide"])
    total_sulfur_dioxide = float(request.form["total_sulfur_dioxide"])
    density = float(request.form["density"])
    ph = float(request.form["ph"])
    sulphates = float(request.form["sulphates"])
    alcohol = float(request.form["alcohol"])
    quality = float(request.form["quality"])

    # Add to numpy array
    predict_data = np.array([[
        fixed_acidity,
        volatile_acidity,
        citric_acid,
        residual_sugar,
        chlorides,
        free_sulfur_dioxide,
        total_sulfur_dioxide,
        density,
        ph,
        sulphates,
        alcohol,
        quality,
    ]])

    # Make and display a prediction based on the form data
    prediction = model.predict(predict_data)
    return render_template(
        "index.html", prediction_text=f"This is {prediction[0]} wine."
    )


if __name__ == "__main__":
    flask_app.run(debug=True)
