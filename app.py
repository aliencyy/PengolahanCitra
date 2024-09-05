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
app.secret_key = 'your_secret_key'  # Kunci rahasia untuk flash messages
app.config['UPLOAD_FOLDER'] = 'src/static/uploads/'  # Simpan gambar di folder static/uploads
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}  # Ekstensi file yang diperbolehkan

# Fungsi untuk memeriksa apakah file memiliki ekstensi yang diizinkan
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/histogram.html', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Cek apakah request berisi file
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        image_type = request.form.get('image_type')

        # Cek apakah tidak ada file yang diunggah
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        # Cek apakah file valid dan diizinkan
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)  # Amankan nama file
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)  # Simpan file di folder yang ditentukan

            # Process the image and create histogram based on type
            img = cv2.imread(file_path)
            hist_img_base64 = None
            equalized_img_base64 = None

            if image_type == 'grayscale':
                # Convert to grayscale and calculate histogram
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])

                # Plot the histogram
                plt.figure()
                plt.plot(hist)
                plt.title('Grayscale Histogram')
                plt.xlabel('Bins')
                plt.ylabel('Number of pixels')
                plt.xlim([0, 256])
                plt.tight_layout()

                # Save histogram plot as an image
                hist_img = BytesIO()
                plt.savefig(hist_img, format='png')
                plt.close()
                hist_img.seek(0)
                hist_img_base64 = base64.b64encode(hist_img.read()).decode('utf-8')

                # Perform histogram equalization
                equalized_img = cv2.equalizeHist(gray_img)
                hist = cv2.calcHist([equalized_img], [0], None, [256], [0, 256])

                # Plot the histogram
                plt.figure()
                plt.plot(hist)
                plt.title('Grayscale Histogram')
                plt.xlabel('Bins')
                plt.ylabel('Number of pixels')
                plt.xlim([0, 256])
                plt.tight_layout()

                # Save histogram plot as an image
                hist_img = BytesIO()
                plt.savefig(hist_img, format='png')
                plt.close()
                hist_img.seek(0)
                equal_hist_img_base64 = base64.b64encode(hist_img.read()).decode('utf-8')

                # Save equalized image
                equalized_img_pil = Image.fromarray(equalized_img)
                equalized_img_io = BytesIO()
                equalized_img_pil.save(equalized_img_io, format='PNG')
                equalized_img_io.seek(0)
                equalized_img_base64 = base64.b64encode(equalized_img_io.read()).decode('utf-8')


            else:
                # Calculate histogram for each channel (RGB)
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

                # Save histogram plot as an image
                hist_img = BytesIO()
                plt.savefig(hist_img, format='png')
                plt.close()
                hist_img.seek(0)
                hist_img_base64 = base64.b64encode(hist_img.read()).decode('utf-8')

                # Perform histogram equalization on each channel
                b, g, r = cv2.split(img)
                b_eq = cv2.equalizeHist(b)
                g_eq = cv2.equalizeHist(g)
                r_eq = cv2.equalizeHist(r)
                result_img = cv2.merge((b_eq, g_eq, r_eq))
                for i, col in enumerate(color):
                    histr = cv2.calcHist([result_img], [i], None, [256], [0, 256])
                    plt.plot(histr, color=col)
                plt.title('Color Equalized Histogram')
                plt.xlabel('Bins')
                plt.ylabel('Number of pixels')
                plt.xlim([0, 256])
                plt.tight_layout()
                #Save histogram plot as an image
                hist_img = BytesIO()
                plt.savefig(hist_img, format='png')
                plt.close()
                hist_img.seek(0)
                equal_hist_img_base64 = base64.b64encode(hist_img.read()).decode('utf-8')

                # Save equalized image
                result_img_pil = Image.fromarray(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
                equalized_img_io = BytesIO()
                result_img_pil.save(equalized_img_io, format='PNG')
                equalized_img_io.seek(0)
                equalized_img_base64 = base64.b64encode(equalized_img_io.read()).decode('utf-8')

            return render_template('histogram.html', filename=filename, hist_img_data=hist_img_base64, equal_hist_img_data=equal_hist_img_base64, equalized_img_data=equalized_img_base64, image_type=image_type)

    # Tampilkan halaman upload
    filename = request.args.get('filename')
    return render_template('histogram.html', filename=filename)

# Route untuk halaman histogram.html
@app.route('/')
def homepage():
    return render_template('homepage.html')

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])  # Buat folder jika belum ada
    app.run(debug=True)
