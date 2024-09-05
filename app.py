import matplotlib
matplotlib.use('Agg')  # Backend non-GUI untuk server

from flask import Flask, render_template, request, flash, redirect, url_for
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from PIL import Image

app = Flask(__name__, static_folder='src', static_url_path='')
app.secret_key = 'your_secret_key'
app.config['UPLOAD_FOLDER'] = 'src/static/uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/histogram.html', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        image_type = request.form.get('image_type')

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            img = cv2.imread(file_path)
            hist_img_base64 = None
            equalized_img_base64 = None
            equal_hist_img_base64 = None

            if image_type == 'grayscale':
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])

                plt.figure()
                plt.plot(hist)
                plt.title('Grayscale Histogram')
                plt.xlabel('Bins')
                plt.ylabel('Number of pixels')
                plt.xlim([0, 256])
                plt.tight_layout()

                hist_img = BytesIO()
                plt.savefig(hist_img, format='png')
                plt.close()
                hist_img.seek(0)
                hist_img_base64 = base64.b64encode(hist_img.read()).decode('utf-8')

                equalized_img = cv2.equalizeHist(gray_img)
                hist = cv2.calcHist([equalized_img], [0], None, [256], [0, 256])

                plt.figure()
                plt.plot(hist)
                plt.title('Equalized Grayscale Histogram')
                plt.xlabel('Bins')
                plt.ylabel('Number of pixels')
                plt.xlim([0, 256])
                plt.tight_layout()

                hist_img = BytesIO()
                plt.savefig(hist_img, format='png')
                plt.close()
                hist_img.seek(0)
                equal_hist_img_base64 = base64.b64encode(hist_img.read()).decode('utf-8')

                equalized_img_pil = Image.fromarray(equalized_img)
                equalized_img_io = BytesIO()
                equalized_img_pil.save(equalized_img_io, format='PNG')
                equalized_img_io.seek(0)
                equalized_img_base64 = base64.b64encode(equalized_img_io.read()).decode('utf-8')

            else:
                color = ('b', 'g', 'r')
                plt.figure()
                for i, col in enumerate(color):
                    histr = cv2.calcHist([img], [i], None, [256], [0, 256])
                    plt.plot(histr, color=col)
                plt.title('Color Histogram')
                plt.xlabel('Bins')
                plt.ylabel('Number of pixels')
                plt.xlim([0, 256])
                plt.tight_layout()

                hist_img = BytesIO()
                plt.savefig(hist_img, format='png')
                plt.close()
                hist_img.seek(0)
                hist_img_base64 = base64.b64encode(hist_img.read()).decode('utf-8')

                b, g, r = cv2.split(img)
                b_eq = cv2.equalizeHist(b)
                g_eq = cv2.equalizeHist(g)
                r_eq = cv2.equalizeHist(r)
                result_img = cv2.merge((b_eq, g_eq, r_eq))

                plt.figure()
                for i, col in enumerate(color):
                    histr = cv2.calcHist([result_img], [i], None, [256], [0, 256])
                    plt.plot(histr, color=col)
                plt.title('Equalized Color Histogram')
                plt.xlabel('Bins')
                plt.ylabel('Number of pixels')
                plt.xlim([0, 256])
                plt.tight_layout()

                hist_img = BytesIO()
                plt.savefig(hist_img, format='png')
                plt.close()
                hist_img.seek(0)
                equal_hist_img_base64 = base64.b64encode(hist_img.read()).decode('utf-8')

                result_img_pil = Image.fromarray(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
                equalized_img_io = BytesIO()
                result_img_pil.save(equalized_img_io, format='PNG')
                equalized_img_io.seek(0)
                equalized_img_base64 = base64.b64encode(equalized_img_io.read()).decode('utf-8')

            return render_template('histogram.html', filename=filename, hist_img_data=hist_img_base64, equal_hist_img_data=equal_hist_img_base64, equalized_img_data=equalized_img_base64, image_type=image_type)

    filename = request.args.get('filename')
    return render_template('histogram.html', filename=filename)

@app.route('/')
def homepage():
    return render_template('homepage.html')

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
