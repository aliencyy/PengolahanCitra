from flask import Flask, render_template, request, flash, redirect, url_for
from werkzeug.utils import secure_filename
import os

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

        # Cek apakah tidak ada file yang diunggah
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        # Cek apakah file valid dan diizinkan
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)  # Amankan nama file
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)  # Simpan file di folder yang ditentukan

            # Redirect ke halaman upload dengan gambar yang sudah diunggah
            return redirect(url_for('upload_file', filename=filename))

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
