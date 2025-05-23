from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
from datetime import datetime
import numpy as np
from tensorflow.keras.models import load_model
from utils.preprocessing import ela_image, srm_average

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

model = load_model('model/best_model_rmsprop.h5')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['image']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Preprocess untuk prediksi
            ela = ela_image(filepath)
            srm = srm_average(filepath)
            ela_input = np.expand_dims(ela / 255.0, axis=0)
            srm_input = np.expand_dims(srm / 255.0, axis=0)

            # === Simpan gambar hasil preprocessing ===
            ela_filename = 'ela_' + filename
            srm_filename = 'srm_' + filename
            ela_path = os.path.join(app.config['UPLOAD_FOLDER'], ela_filename)
            srm_path = os.path.join(app.config['UPLOAD_FOLDER'], srm_filename)

            import cv2
            cv2.imwrite(ela_path, ela)
            cv2.imwrite(srm_path, srm)

            # Prediksi
            prediction = model.predict([ela_input, srm_input])[0]
            label = 'Dimanipulasi' if np.argmax(prediction) == 1 else 'Asli'
            confidence = float(np.max(prediction) * 100)

            return render_template(
                'result.html',
                filename=filename,
                ela_filename=ela_filename,
                srm_filename=srm_filename,
                label=label,
                confidence=confidence
            )
    return render_template('upload.html')

@app.route('/history')
def history():
    return render_template('history.html')

@app.route('/uploads/<filename>')
def send_file(filename):
    return redirect(url_for('static', filename='uploads/' + filename))

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
