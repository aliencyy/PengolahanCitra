import matplotlib
matplotlib.use('Agg')  # Backend non-GUI untuk server

from flask import Flask, jsonify, render_template, request, flash, redirect, url_for
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
            return redirect(request.url),jsonify({'error': 'No file part'}), 400
        file = request.files['file']
        image_type = request.form.get('image_type')

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url),jsonify({'error': 'No selected file'}), 400

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
            blurred_background_img_base64 = None
            segmentation_img_base64 = None

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
                slider_value = int(request.form.get('blurRange', 50))  # Default to 50 if not provided

                # Define min and max kernel sizes
                min_kernel_size = 0
                max_kernel_size = 31

                # Convert slider percentage (0 - 100) to kernel size
                blur_value = min_kernel_size + (slider_value / 100) * (max_kernel_size - min_kernel_size)
                blur_value = int(blur_value)  # Convert to integer
                if blur_value % 2 == 0:
                    blur_value += 1  # Ensure it's odd

                face_cascade = cv2.CascadeClassifier('src/static/models/haarcascade_frontalface_default.xml')
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                if len(faces) > 0:
                    for (x, y, w, h) in faces:
                        face = img[y:y+h, x:x+w]
                        blurred_face = cv2.GaussianBlur(face, (blur_value, blur_value), 2)
                        img[y:y+h, x:x+w] = blurred_face

                # Convert the result to base64
                img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                img_io = BytesIO()
                img_pil.save(img_io, format='PNG')
                img_io.seek(0)
                face_blur_img_base64 = base64.b64encode(img_io.read()).decode('utf-8')

            elif image_type == 'background_blurring':
                # Create a mask for GrabCut
                mask = np.zeros(img.shape[:2], np.uint8)
                
                # Define a rectangle for the initial GrabCut segmentation
                # Adjust the rectangle (x, y, width, height) based on your image
                # Coba optimalkan ukuran rect
                rect = (int(img.shape[1] * 0.1), int(img.shape[0] * 0.1), 
                        int(img.shape[1] * 0.8), int(img.shape[0] * 0.8))

                # Create a mask for the background and foreground
                bgd_model = np.zeros((1, 65), np.float64)
                fgd_model = np.zeros((1, 65), np.float64)
                
                # Apply GrabCut algorithm
                cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 7, cv2.GC_INIT_WITH_RECT)
                
                # Create a binary mask where foreground pixels are marked with 1
                mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
                
                # Blur the entire image
                blurred_img = cv2.GaussianBlur(img, (21, 21), 0)

                # Use the mask to keep the object in focus and blur the background
                object_in_focus = img * mask2[:, :, np.newaxis]
                background_blurred = blurred_img * (1 - mask2[:, :, np.newaxis])

                # Combine the focused object and blurred background
                final_img = object_in_focus + background_blurred

                # Convert the segmentation result to base64 for rendering in HTML
                segmentation_img_pil = Image.fromarray(cv2.cvtColor(object_in_focus, cv2.COLOR_BGR2RGB))
                segmentation_img_io = BytesIO()
                segmentation_img_pil.save(segmentation_img_io, format='PNG')
                segmentation_img_io.seek(0)
                segmentation_img_base64 = base64.b64encode(segmentation_img_io.read()).decode('utf-8')

                # Convert the final blurred background image to base64 for rendering in HTML
                final_img_pil = Image.fromarray(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB))
                final_img_io = BytesIO()
                final_img_pil.save(final_img_io, format='PNG')
                final_img_io.seek(0)
                blurred_background_img_base64 = base64.b64encode(final_img_io.read()).decode('utf-8')  

            return render_template('process.html', filename=filename, hist_img_data=hist_img_base64, equal_hist_img_data=equal_hist_img_base64, equalized_img_data=equalized_img_base64, edge_img_data=edge_img_base64, face_img_data=face_img_base64, face_blur_img_data=face_blur_img_base64 ,segmentation_img_data=segmentation_img_base64, blurred_background_img_data=blurred_background_img_base64,image_type=image_type)


    filename = request.args.get('filename')
    return render_template('process.html', filename=filename)

# Route untuk memperbarui tingkat blur
@app.route('/update_blur', methods=['POST'])
def update_blur():
    filename = request.form.get('filename')
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # Ensure the file exists
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404

    # Load the image
    img = cv2.imread(file_path)

    # Get the slider value
    slider_value = int(request.form.get('blurRange', 50))  # Default to 50 if not provided

    # Define min and max kernel sizes
    min_kernel_size = 1
    max_kernel_size = 31

    # Convert slider percentage (0 - 100) to kernel size
    blur_value = min_kernel_size + (slider_value / 100) * (max_kernel_size - min_kernel_size)
    blur_value = int(blur_value)
    if blur_value % 2 == 0:
        blur_value += 1  # Ensure it's odd

    # Detect faces and apply blur
    face_cascade = cv2.CascadeClassifier('src/static/models/haarcascade_frontalface_default.xml')
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            face = img[y:y+h, x:x+w]
            blurred_face = cv2.GaussianBlur(face, (blur_value, blur_value), 0)
            img[y:y+h, x:x+w] = blurred_face

    # Convert the result to base64
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img_io = BytesIO()
    img_pil.save(img_io, format='PNG')
    img_io.seek(0)
    face_blur_img_base64 = base64.b64encode(img_io.read()).decode('utf-8')

    return jsonify({'updated_image': face_blur_img_base64})

@app.route('/')
def homepage():
    return render_template('homepage.html')

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
