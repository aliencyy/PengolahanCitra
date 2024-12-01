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
from scipy import ndimage
import networkx as nx
from collections import Counter
import heapq

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
        interpolation_method = request.form.get('interpolation')
        scale_factor = float(request.form.get('scale_factor', 1.0))

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
            vintage_sepia_img_base64 = None
            harris_corner_img_base64 = None
            morphed_img_base64 = None
            scaled_images_data = [] 
            morph_description = None
            noise_img_base64 = None
            restored_img_base64 = None

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

            elif image_type == 'vintage_sepia':
                # Convert the image to sepia tone
                img_sepia = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_sepia = cv2.transform(img_sepia, np.matrix([[0.393, 0.769, 0.189],
                                                                [0.349, 0.686, 0.168],
                                                                [0.272, 0.534, 0.131]]))

                # Add noise to enhance the vintage effect
                # Generate Gaussian noise with lower intensity
                noise = np.random.normal(0, 10, img_sepia.shape)  # mean=0, std=10 for lighter noise
                noisy_img = img_sepia + noise

                # Clip values to keep them within valid pixel range
                noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)

                # Convert the result to base64
                img_io = BytesIO()
                img_pil = Image.fromarray(noisy_img)
                img_pil.save(img_io, format='PNG')
                img_io.seek(0)
                vintage_sepia_img_base64 = base64.b64encode(img_io.read()).decode('utf-8')

            elif image_type == 'harris_corner':
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray_img = np.float32(gray_img)

                # Harris corner detection
                dst = cv2.cornerHarris(gray_img, 2, 3, 0.04)

                # Result is dilated for marking the corners
                dst = cv2.dilate(dst, None)

                # Thresholding to mark corners
                img[dst > 0.01 * dst.max()] = [0, 0, 255]  # Mark corners in red

                # Convert the result to base64
                img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                img_io = BytesIO()
                img_pil.save(img_io, format='PNG')
                img_io.seek(0)
                harris_corner_img_base64 = base64.b64encode(img_io.read()).decode('utf-8')

            elif image_type == 'perspective':
                # Transformasi perspektif
                flash('Silakan pilih 4 titik pada gambar untuk melakukan transformasi perspektif.')
                return render_template('process.html', filename=filename, image_type=image_type)

            elif image_type == 'morphology':
                morphology_type = request.form.get('morphology_type')
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                kernel = np.ones((5, 5), np.uint8)  # Kernel ukuran 5x5 untuk operasi morphology

                if morphology_type == 'dilation':
                    morphed_img = cv2.dilate(gray_img, kernel, iterations=2)
                    morph_description = "Gambar setelah dilation:"
                elif morphology_type == 'erosion':
                    morphed_img = cv2.erode(gray_img, kernel, iterations=2)
                    morph_description = "Gambar setelah erosion:"
                elif morphology_type == 'opening':
                    morphed_img = cv2.morphologyEx(gray_img, cv2.MORPH_OPEN, kernel, iterations=2)
                    morph_description = "Gambar setelah opening:"
                elif morphology_type == 'closing':
                    morphed_img = cv2.morphologyEx(gray_img, cv2.MORPH_CLOSE, kernel, iterations=3)
                    morph_description = "Gambar setelah closing:"
                else:
                    morph_description = "Invalid morphological operation."
                
                # Konversi gambar morfologi menjadi base64 untuk ditampilkan
                morphed_img_pil = Image.fromarray(morphed_img)
                morphed_img_io = BytesIO()
                morphed_img_pil.save(morphed_img_io, format='PNG')
                morphed_img_io.seek(0)
                morphed_img_base64 = base64.b64encode(morphed_img_io.read()).decode('utf-8')

            elif image_type == 'scaling':
                interpolation = {
                    'nearest': cv2.INTER_NEAREST,
                    'bilinear': cv2.INTER_LINEAR,
                    'bicubic': cv2.INTER_CUBIC,
                }
                resized_images = []
                interpolation_method = request.form.get('interpolation')
                scale_factor = float(request.form.get('scale_factor'))
                
                if interpolation_method == 'compare_all':
                    for method_name, interp_type in interpolation.items():
                        scaled_img = cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=interp_type)
                        resized_images.append((method_name, scaled_img))
                else:
                    interp_type = interpolation.get(interpolation_method, cv2.INTER_LINEAR)
                    scaled_img = cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=interp_type)
                    resized_images.append((interpolation_method, scaled_img))

                # Convert resized images to base64 for HTML display
                scaled_images_data = []
                for method_name, resized_img in resized_images:
                    _, buffer = cv2.imencode('.png', resized_img)
                    img_base64 = base64.b64encode(buffer).decode('utf-8')
                    scaled_images_data.append((method_name, img_base64))
            
            # Menangani jenis restorasi
            if image_type == 'restoration':
                restoration_noise = request.form.get('restoration_noise')
                restoration_type = request.form.get('restoration_type')

                # Tambahkan noise sesuai pilihan
                if restoration_noise == 'salt_and_pepper_noise':
                    noise = np.random.choice([0, 255], img.shape, p=[0.95, 0.05]).astype(np.uint8)
                    img = cv2.add(img, noise)

                elif restoration_noise == 'gaussian_noise':
                    gaussian_noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
                    img = cv2.add(img, gaussian_noise)

                elif restoration_noise == 'speckle_noise':
                    speckle_noise = np.random.randn(*img.shape) * 25
                    img = img + img * speckle_noise

                elif restoration_noise == 'periodic_noise':
                    rows, cols, _ = img.shape
                    x = np.linspace(0, 2 * np.pi, cols)
                    y = np.linspace(0, 2 * np.pi, rows)
                    X, Y = np.meshgrid(x, y)
                    periodic_noise = (np.sin(X * 10) + np.sin(Y * 10)) * 25
                    img = cv2.add(img, periodic_noise.astype(np.uint8))

                # Konversi gambar hasil ke base64 untuk ditampilkan
                img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                img_io = BytesIO()
                img_pil.save(img_io, format='PNG')
                img_io.seek(0)
                noise_img_base64 = base64.b64encode(img_io.read()).decode('utf-8')

                # Terapkan restorasi sesuai pilihan
                if restoration_type == 'lowpass_filtering':
                    img = cv2.GaussianBlur(img, (5, 5), 0)

                elif restoration_type == 'median_filtering':
                    img = cv2.medianBlur(img, 5)

                elif restoration_type == 'rankorder_filtering':
                    # Menggunakan median filter sebagai contoh
                    img = cv2.medianBlur(img, 5)

                elif restoration_type == 'outlier_method':
                    # Contoh sederhana menggunakan bilateral filter
                    img = cv2.bilateralFilter(img, 9, 75, 75)

                # Konversi gambar hasil ke base64 untuk ditampilkan
                img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                img_io = BytesIO()
                img_pil.save(img_io, format='PNG')
                img_io.seek(0)
                restored_img_base64 = base64.b64encode(img_io.read()).decode('utf-8')
            return render_template('process.html', filename=filename, hist_img_data=hist_img_base64, equal_hist_img_data=equal_hist_img_base64, equalized_img_data=equalized_img_base64, edge_img_data=edge_img_base64, face_img_data=face_img_base64, face_blur_img_data=face_blur_img_base64 ,segmentation_img_data=segmentation_img_base64, blurred_background_img_data=blurred_background_img_base64, vintage_sepia_img_data=vintage_sepia_img_base64, harris_corner_img_data=harris_corner_img_base64,morph_description=morph_description, morphed_img_data=morphed_img_base64,scaling_images=scaled_images_data, noise_img_data= noise_img_base64, restored_img_data=restored_img_base64, image_type=image_type)

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

@app.route('/perspective_transform', methods=['POST'])
def perspective_transform():
    data = request.get_json()
    filename = data.get('filename')
    points = data.get('points')

    if filename and points:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        img = cv2.imread(file_path)

        # Cek ukuran gambar
        img_height, img_width = img.shape[:2]
        if img_height < 200 or img_width < 200:
            return jsonify({'error': 'Gambar terlalu kecil, silakan gunakan gambar dengan resolusi yang lebih tinggi.'}), 400

        # Points from the user (source points)
        src_points = np.float32([[points[0], points[1]],  # Kiri Atas
                                  [points[2], points[3]],  # Kanan Atas
                                  [points[4], points[5]],  # Kanan Bawah
                                  [points[6], points[7]]])  # Kiri Bawah

        # Hitung lebar dan tinggi baru dari titik sumber
        width = int(max(np.linalg.norm(src_points[0] - src_points[1]), np.linalg.norm(src_points[2] - src_points[3])))
        height = int(max(np.linalg.norm(src_points[0] - src_points[3]), np.linalg.norm(src_points[1] - src_points[2])))

        # Batasan maksimum dan minimum
        max_width = img.shape[1]  # Lebar gambar asli
        max_height = img.shape[0]  # Tinggi gambar asli

        # Batasi width dan height agar tidak melebihi ukuran gambar asli
        width = min(width, max_width)
        height = min(height, max_height)

        # Tentukan batasan minimum
        min_width, min_height = 100, 100
        width = max(min(width, img.shape[1]), min_width)
        height = max(min(height, img.shape[0]), min_height)

        # Tentukan dst_points dengan ukuran baru
        dst_points = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

        # Compute the perspective transform matrix
        M = cv2.getPerspectiveTransform(src_points, dst_points)

        # Apply the perspective transform to the image
        transformed_img = cv2.warpPerspective(img, M, (width, height))

        # Convert the transformed image to base64 to send back to the frontend
        _, buffer = cv2.imencode('.png', transformed_img)
        transformed_img_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({'transformed_img_data': transformed_img_base64})

    return jsonify({'error': 'Invalid data'}), 400

class Node:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq


# Build Huffman Tree
def build_huffman_tree(data):
    if not data:
        return None

    frequency = Counter(data)
    heap = [Node(char, freq) for char, freq in frequency.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = Node(None, left.freq + right.freq)
        merged.left = left
        merged.right = right
        heapq.heappush(heap, merged)

    return heap[0]  # Root of the tree


# Generate Huffman Codes
def generate_huffman_codes(root, current_code="", codes={}):
    if root is None:
        return

    if root.char is not None:  # Leaf node
        codes[root.char] = current_code

    generate_huffman_codes(root.left, current_code + "0", codes)
    generate_huffman_codes(root.right, current_code + "1", codes)

    return codes


# Draw Huffman Tree with matplotlib and networkx
def draw_huffman_tree(root):
    def add_edges(graph, node, parent=None, label=""):
        if node:
            node_label = f"{node.char} ({node.freq})" if node.char else f"{node.freq}"
            graph.add_node(id(node), label=node_label)

            if parent:
                graph.add_edge(id(parent), id(node), label=label)

            add_edges(graph, node.left, node, "0")
            add_edges(graph, node.right, node, "1")

    def hierarchy_pos(G, root, width=1.0, vert_gap=0.2, vert_loc=0, xcenter=0.5):
        """
        Generate a hierarchical layout for tree-like graphs.
        """
        pos = {root: (xcenter, vert_loc)}
        children = list(G.successors(root))
        if not children:
            return pos
        dx = width / len(children)
        nextx = xcenter - width / 2 - dx / 2
        for child in children:
            nextx += dx
            pos.update(
                hierarchy_pos(
                    G, child, width=dx, vert_gap=vert_gap, vert_loc=vert_loc - vert_gap, xcenter=nextx
                )
            )
        return pos

    G = nx.DiGraph()
    add_edges(G, root)

    # Posisi node dalam tata letak hierarkis
    pos = hierarchy_pos(G, id(root))
    labels = nx.get_node_attributes(G, 'label')
    edge_labels = nx.get_edge_attributes(G, 'label')

    # Plot graf menggunakan matplotlib
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=False, node_size=2000, node_color='skyblue', font_size=10, arrows=True)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=12, font_color='black')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)
    plt.title("Huffman Tree")
    plt.axis("off")

    # Simpan grafik ke file
    tree_image_path = os.path.join(app.config['UPLOAD_FOLDER'], "huffman_tree.png")
    plt.savefig(tree_image_path, format="png")
    plt.close()

    return "static/uploads/huffman_tree.png"


# Huffman route
@app.route('/huffman', methods=['GET', 'POST'])
def huffman():
    if request.method == 'POST':
        text_to_compress = request.form.get('text_to_compress', '')
        if not text_to_compress:
            flash("Teks tidak boleh kosong!")
            return redirect(url_for('huffman'))

        # Huffman Coding
        root = build_huffman_tree(text_to_compress)
        huffman_dict = generate_huffman_codes(root)
        encoded_data = "".join(huffman_dict[char] for char in text_to_compress)
        decoded_data = text_to_compress  # Decoded langsung sama dengan input

        # Generate Huffman Tree Image
        huffman_tree_image = draw_huffman_tree(root)

        return render_template(
            'huffman.html',
            original_text=text_to_compress,
            encoded_data=encoded_data,
            decoded_data=decoded_data,
            huffman_dict=huffman_dict,
            huffman_tree_image=huffman_tree_image
        )

    return render_template('huffman.html')


def add_noise(image, noise_type):
    """Tambahkan noise ke gambar"""
    if noise_type == 'salt_and_pepper':  
        # Salt and Pepper Noise menggunakan metode impulse noise  
        # Buat noise dengan ukuran sama dengan gambar asli  
        impulse_noise = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8) 
        
        # Gunakan threshold untuk membuat noise biner  
        # Anda bisa sesuaikan parameter threshold  
        ret, impulse_noise = cv2.threshold(  
            np.random.randint(0, 256, image.shape, dtype=np.uint8),   
            250,  # Threshold tinggi   
            255,  # Nilai maksimum  
            cv2.THRESH_BINARY  
        )  
        
        # Kurangi intensitas noise  
        impulse_noise = (impulse_noise * 0.5).astype(np.uint8)  
        
        # Tambahkan noise ke gambar asli  
        noisy_image = cv2.add(image, impulse_noise)  
        
        return noisy_image 
    
    if noise_type == 'gaussian':  
        # Buat array kosong dengan ukuran sama persis dengan gambar  
        # Ini akan menjadi tempat kita membuat noise  
        noise = np.zeros(image.shape[:2], dtype=np.uint8)  
        
        # Membuat noise Gaussian menggunakan cv2.randn()  
        # Parameter:  
        # - noise: array yang akan diisi noise  
        # - 255: mean (pusat distribusi noise)  
        # - 150: standar deviasi (sebaran noise)  
        cv2.randn(noise, 255, 150)  
        
        # Kurangi intensitas noise dengan mengalikan 0.5  
        # Ini membuat noise tidak terlalu kuat  
        noise = (noise * 0.5).astype(np.uint8)  
        
        # Untuk gambar berwarna (RGB)  
        if len(image.shape) == 3:  
            # Buat salinan gambar  
            noisy_image = image.copy()  
            
            # Tambahkan noise ke setiap channel warna  
            for i in range(image.shape[2]):  
                noisy_image[:,:,i] = cv2.add(noisy_image[:,:,i], noise)  
            
            return noisy_image  
        
        # Untuk gambar hitam putih (single channel)  
        else:  
            # Tambahkan noise langsung ke gambar  
            return cv2.add(image, noise)  
    
    if noise_type == 'speckle':  
        # Implementasi Speckle Noise yang lebih komprehensif  
        
        # Untuk gambar berwarna (multi-channel)  
        if len(image.shape) == 3:  
            noisy_image = image.copy().astype(np.float32)  
            
            # Generate noise untuk setiap channel  
            for i in range(image.shape[2]):  
                # Buat noise Gaussian dengan mean 0 dan standar deviasi 0.1  
                noise = np.random.normal(0, 0.1, image.shape[:2])  
                
                # Terapkan noise multiplicative  
                # Pixel baru = Pixel asli * (1 + noise)  
                noisy_image[:,:,i] = noisy_image[:,:,i] * (1 + noise)  
            
            # Clip nilai antara 0-255 dan kembalikan ke uint8  
            return np.clip(noisy_image, 0, 255).astype(np.uint8)  
        
        # Untuk gambar hitam putih (single channel)  
        else:  
            # Konversi ke float32 untuk perhitungan  
            noisy_image = image.astype(np.float32)  
            
            # Buat noise Gaussian dengan mean 0 dan standar deviasi 0.1  
            noise = np.random.normal(0, 0.1, image.shape)  
            
            # Terapkan noise multiplicative  
            # Pixel baru = Pixel asli * (1 + noise)  
            noisy_image = noisy_image * (1 + noise)  
            
            # Clip nilai antara 0-255 dan kembalikan ke uint8  
            return np.clip(noisy_image, 0, 255).astype(np.uint8)  
    
    if noise_type == 'periodic':  
        # Untuk gambar berwarna (multi-channel)  
        if len(image.shape) == 3:  
            noisy_image = image.copy().astype(np.float32)  
            
            # Generate periodic noise untuk setiap channel  
            for channel in range(image.shape[2]):  
                # Buat noise periodic dengan beberapa frekuensi  
                noise = np.zeros_like(image[:,:,channel], dtype=np.float32)  
                
                # Beberapa variasi frekuensi untuk mensimulasikan interferensi  
                frequencies = [  
                    (32, 50),   # Frekuensi horizontal  
                    (64, 30),   # Frekuensi vertikal  
                    (16, 40)    # Frekuensi diagonal  
                ]  
                
                for freq, amplitude in frequencies:  
                    for i in range(noise.shape[0]):  
                        for j in range(noise.shape[1]):  
                            # Kombinasi fungsi sinusoidal  
                            noise[i, j] += amplitude * np.sin(2 * np.pi * j / freq)  
                
                # Tambahkan noise ke channel  
                noisy_image[:,:,channel] = cv2.add(  
                    noisy_image[:,:,channel],   
                    noise  
                )  
            
            # Clip dan kembalikan ke uint8  
            return np.clip(noisy_image, 0, 255).astype(np.uint8)  
        
        # Untuk gambar hitam putih (single channel)  
        else:  
            # Konversi ke float32  
            noisy_image = image.astype(np.float32)  
            
            # Buat noise periodic  
            noise = np.zeros_like(image, dtype=np.float32)  
            
            # Beberapa variasi frekuensi untuk mensimulasikan interferensi  
            frequencies = [  
                (32, 50),   # Frekuensi horizontal  
                (64, 30),   # Frekuensi vertikal  
                (16, 40)    # Frekuensi diagonal  
            ]  
            
            for freq, amplitude in frequencies:  
                for i in range(noise.shape[0]):  
                    for j in range(noise.shape[1]):  
                        # Kombinasi fungsi sinusoidal  
                        noise[i, j] += amplitude * np.sin(2 * np.pi * j / freq)  
            
            # Tambahkan noise ke gambar  
            noisy_image = cv2.add(noisy_image, noise)  
            
            # Clip dan kembalikan ke uint8  
            return np.clip(noisy_image, 0, 255).astype(np.uint8)  
    
def restore_image(noisy_image, restoration_type):  
    """Pulihkan gambar dengan metode tertentu"""  
    if restoration_type == 'lowpass':  
        # Lowpass Filtering menggunakan uniform filter  
        
        # Untuk gambar berwarna (multi-channel)  
        if len(noisy_image.shape) == 3:  
            # Buat salinan gambar  
            restored_image = noisy_image.copy()  
            
            # Proses setiap channel  
            for i in range(noisy_image.shape[2]):  
                # Terapkan uniform filter  
                restored_image[:,:,i] = ndimage.uniform_filter(  
                    noisy_image[:,:,i],   
                    size=7  # Ukuran filter, bisa disesuaikan  
                )  
            
            return restored_image  
        
        # Untuk gambar hitam putih (single channel)  
        else:  
            # Terapkan uniform filter  
            return ndimage.uniform_filter(noisy_image, size=7)   
    
    elif restoration_type == 'median':  
        # Median Filtering  
        return cv2.medianBlur(noisy_image, 7)  
    
    if restoration_type == 'rank_order':  
        # Rank-Order Filtering menggunakan median filter dengan footprint khusus  
        
        # Definisikan footprint cross  
        cross = np.array([  
            [0, 1, 0],  
            [1, 1, 1],  
            [0, 1, 0]  
        ])  
        
        # Untuk gambar berwarna (multi-channel)  
        if len(noisy_image.shape) == 3:  
            # Buat salinan gambar  
            restored_image = noisy_image.copy()  
            
            # Proses setiap channel  
            for i in range(noisy_image.shape[2]):  
                # Terapkan median filter dengan footprint cross  
                restored_image[:,:,i] = ndimage.median_filter(  
                    noisy_image[:,:,i],   
                    footprint=cross  
                )  
            
            return restored_image  
        
        # Untuk gambar hitam putih (single channel)  
        else:  
            # Terapkan median filter dengan footprint cross  
            return ndimage.median_filter(noisy_image, footprint=cross)  
    
    if restoration_type == 'outlier':  
        # Definisikan kernel average  
        av = np.array([  
            [0, 1, 0],  
            [1, 1, 1],  
            [0, 1, 0]  
        ]) / 8.0  
        
        # Threshold untuk mendeteksi outlier  
        D = 0.2  
        
        # Untuk gambar berwarna (multi-channel)  
        if len(noisy_image.shape) == 3:  
            # Buat salinan gambar  
            restored_image = noisy_image.copy().astype(np.float32) / 255.0  
            restored_channels = []  
            
            # Proses setiap channel  
            for channel in cv2.split(restored_image):  
                # Konvolusi dengan kernel average  
                image_sp_av = ndimage.convolve(channel, av)  
                
                # Deteksi outlier  
                r = (np.abs(channel - image_sp_av) > D).astype(float)  
                
                # Rekonstruksi gambar  
                restored_channel = r * image_sp_av + (1 - r) * channel  
                restored_channels.append(restored_channel)  
            
            # Gabungkan channel dan kembalikan ke uint8  
            restored_image = cv2.merge(restored_channels)  
            return (restored_image * 255).astype(np.uint8)  
        
        # Untuk gambar hitam putih (single channel)  
        else:  
            # Konversi ke float  
            channel = noisy_image.astype(np.float32) / 255.0  
            
            # Konvolusi dengan kernel average  
            image_sp_av = ndimage.convolve(channel, av)  
            
            # Deteksi outlier  
            r = (np.abs(channel - image_sp_av) > D).astype(float)  
            
            # Rekonstruksi gambar  
            restored_channel = r * image_sp_av + (1 - r) * channel  
            
            # Kembalikan ke uint8  
            return (restored_channel * 255).astype(np.uint8)  

@app.route('/restoration', methods=['GET', 'POST'])
def restoration():
    if request.method == 'POST':
        uploaded_file = request.files.get('image')

        if uploaded_file is None or uploaded_file.filename == '':
            flash('Tidak ada file yang diunggah', 'error')
            return redirect(request.url)

        if uploaded_file and allowed_file(uploaded_file.filename):
            filename = secure_filename(uploaded_file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            uploaded_file.save(file_path)

            # Membaca gambar
            img = cv2.imread(file_path)
            if img is None:
                flash('Gambar tidak berhasil dibaca', 'error')
                return redirect(request.url)

            # Melakukan pengolahan (contoh: blur)
            processed_img = cv2.GaussianBlur(img, (5, 5), 0)
            if processed_img is None or processed_img.size == 0:
                flash('Proses gambar gagal', 'error')
                return redirect(request.url)

            # Menyimpan gambar yang telah diproses
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_' + filename)
            cv2.imwrite(output_path, processed_img)

            # Menampilkan gambar yang telah diproses
            return render_template('restoration.html', filename=filename, output_filename='processed_' + filename)

        else:
            flash('Jenis file tidak diperbolehkan', 'error')
            return redirect(request.url)

    return render_template('restoration.html',original_image=f'uploads/original_{filename}',
                                   noisy_image=f'uploads/noisy_{filename}', 
                                   restored_image=f'uploads/restored_{filename}')

@app.route('/')
def homepage():
    return render_template('homepage.html')

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
