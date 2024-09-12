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

@app.route('/process.html', methods=['GET', 'POST'])
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
            edge_img_base64 = None  # Variable for edge detection image
            face_img_base64 = None  # Variable for face detection image
            face_blur_img_base64 = None  # Variable for face blurring image

            if image_type == 'grayscale':
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])

                # Histogram
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

                # Equalized Image
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

            elif image_type == 'rgb':
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

                # Equalized Image
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

            elif image_type == 'edge_detection':
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray_img, 100, 200)

                edge_img_pil = Image.fromarray(edges)
                edge_img_io = BytesIO()
                edge_img_pil.save(edge_img_io, format='PNG')
                edge_img_io.seek(0)
                edge_img_base64 = base64.b64encode(edge_img_io.read()).decode('utf-8')

            elif image_type == 'face_detection':
                face_cascade = cv2.CascadeClassifier('src/static/models/haarcascade_frontalface_default.xml')
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                # Draw rectangles around the faces
                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

                # Convert the result to base64
                img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                img_io = BytesIO()
                img_pil.save(img_io, format='PNG')
                img_io.seek(0)
                face_img_base64 = base64.b64encode(img_io.read()).decode('utf-8')
            
            elif image_type == 'face_blurring':
                face_cascade = cv2.CascadeClassifier('src/static/models/haarcascade_frontalface_default.xml')
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                # Blur faces detected in the image
                if len(faces) > 0:
                    for (x, y, w, h) in faces:
                        # Get the face region
                        face = img[y:y+h, x:x+w]

                        # Apply Gaussian blur to the face region
                        blurred_face = cv2.GaussianBlur(face, (99, 99), 30)

                        # Replace the original face with blurred face
                        img[y:y+h, x:x+w] = blurred_face

                        # Convert the result to base64
                        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                        img_io = BytesIO()
                        img_pil.save(img_io, format='PNG')
                        img_io.seek(0)
                        face_blur_img_base64 = base64.b64encode(img_io.read()).decode('utf-8')


            return render_template('process.html', filename=filename, hist_img_data=hist_img_base64, equal_hist_img_data=equal_hist_img_base64, equalized_img_data=equalized_img_base64, edge_img_data=edge_img_base64, face_img_data=face_img_base64, face_blur_img_data=face_blur_img_base64 , image_type=image_type)

    filename = request.args.get('filename')
    return render_template('process.html', filename=filename)

@app.route('/')
def homepage():
    return render_template('homepage.html')

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
