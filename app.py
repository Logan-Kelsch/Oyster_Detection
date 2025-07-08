# app.py
import os
from flask import (
    Flask, render_template, request,
    redirect, url_for, send_file, Response
)
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
    in_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    out_path = os.path.join(app.config['OUTPUT_FOLDER'], 'anno_'+filename)
    f.save(in_path)

    # call your script’s function
    anno_img.run(in_path, out_path)

    # return the annotated image
    return send_file(out_path, mimetype='image/jpeg')

@app.route('/detect_video', methods=['POST'])
def detect_video():
    f = request.files['video']
    filename = secure_filename(f.filename)
    in_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    out_path = os.path.join(app.config['OUTPUT_FOLDER'], 'anno_'+filename)
    f.save(in_path)

    # call your video annotation
    anno_vid.run(in_path, out_path)

    # send back the processed video
    return send_file(out_path, mimetype='video/mp4')

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
