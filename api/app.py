import json
import time
from io import BytesIO
import os
from flask import Flask, request, jsonify, send_file, render_template
from file_func import upload_image
from tiles import voronoi_with_region_merging  

#template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../templates"))

app = Flask(__name__)

@app.route('/')
def index():
    """
    Serve the main HTML page for image upload.
    """
    print ("../templates/upload.html")
    return render_template("D:/Projects/tile-mosaic/templates/upload.html")

@app.route('/api/test')
def test():
    """
    Serve the test endpoint.
    """

    return "This is test endpoint."

@app.route('/api/voronoi', methods=['POST'])
def render_voronoi():
    """
    Add docs for this later. Probably going to change a lot..
    """
    try:
        input_image = upload_image()
        # Apply the Voronoi mosaic function
        voronoi_image, merged_image = voronoi_with_region_merging(input_image, num_points=1000, line_thickness=2, num_neighbors=5)
        
        # Convert the processed image to bytes for sending as a response
        img_io = BytesIO()
        merged_image.save(img_io, format="JPEG")
        img_io.seek(0)  # Reset stream pointer to the beginning
        
        # Return the processed image file
        return send_file(img_io, mimetype='image/jpeg', as_attachment=True, download_name="processed_image.jpg")
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def main():
    app.run(port=5000, debug=False)
    print ("Server is running now.")

if __name__ == "__main__":

    main()