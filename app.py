from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
from disease_database import disease_info
from werkzeug.utils import secure_filename

app = Flask(__name__)

# ===============================
# Configuration
# ===============================
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load trained model
model = load_model("final_eye_disease_model.h5")

class_names = [
    "cataract",
    "diabetic_retinopathy",
    "glaucoma",
    "normal",
    "retina_disease"
]

# ===============================
# Main Route (Handles GET & POST)
# ===============================
@app.route("/", methods=["GET", "POST"])
def home():

    # If user just opens website
    if request.method == "GET":
        return render_template("upload.html")

    # If user uploads file
    if request.method == "POST":

        if "file" not in request.files:
            return render_template("upload.html")

        file = request.files["file"]

        if file.filename == "":
            return render_template("upload.html")

        # Secure filename
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # Preprocess image
        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions)
        predicted_class = class_names[predicted_index]

        confidence = round(float(np.max(predictions)) * 100, 2)

        info = disease_info.get(predicted_class, {})

        return render_template(
            "result.html",
            prediction=predicted_class.replace("_", " ").title(),
            confidence=confidence,
            info=info,
            image_path=filepath
        )


# ===============================
# Run App
# ===============================
if __name__ == "__main__":
    app.run(debug=True)