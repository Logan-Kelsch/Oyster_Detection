# app.py
import os
from io import BytesIO

import cv2
import numpy as np
from flask import (
    Flask, render_template, request,
    redirect, url_for, send_file, Response
)
from ultralytics import YOLO
from werkzeug.utils import secure_filename

import anno_img
import anno_vid
import anno_liv

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

@app.route('/')
def root():
    return redirect(url_for('login'))

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/index.html')
def dashboard():
    return render_template('index.html')

@app.route('/detect_image', methods=['POST'])
def detect_image():
    f = request.files['image']
    filename = secure_filename(f.filename)

    if not anno_img.is_allowed_file(filename):
        return "Unsupported file type.", 400

    ext = filename.rsplit('.', 1)[1].lower()
    image_bytes = f.read()

    annotated_io, mime_type = anno_img.annotate_image_bytes(image_bytes, ext, conf_threshold=0.75)


    return send_file(
        annotated_io,
        mimetype=mime_type,
        as_attachment=True,
        download_name=f'annotated.{ext}'
    )

@app.route('/detect_video', methods=['POST'])
def detect_video():
    f = request.files['video']
    video_bytes = f.read()

    from anno_vid import annotate_video_bytes
    annotated_io, mime_type = annotate_video_bytes(video_bytes)

    return send_file(
        annotated_io,
        mimetype=mime_type,
        as_attachment=True,
        download_name='annotated_video.mp4'
    )

@app.route('/livestream')
def livestream_page():
    return render_template('livestream.html')

@app.route('/video_feed')
def video_feed():
    # uses the generator from anno_liv
    return Response(
        anno_liv.frame_generator(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

if __name__ == "__main__":
    app.run(debug=True)
