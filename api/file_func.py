from PIL import Image
from io import BytesIO
from flask import Flask, request, jsonify, send_file

def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Open the image directly from the uploaded file (in-memory)
        input_image = Image.open(file.stream).convert("RGB")
        return input_image
    
    except Exception as e:
        return e