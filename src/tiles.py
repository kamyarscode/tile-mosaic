
import numpy as np
from scipy.spatial import Voronoi
from PIL import Image, ImageDraw

# Attempt to do this tile using voronoi transformation method
def voronoi_mosaic(image, num_points=500, line_thickness=1):
    """
    Applies a Voronoi transformation to an image to create a mosaic effect.

    Args:
        image (PIL.Image.Image): The input image as a Pillow object.
        num_points (int): Number of seed points for the Voronoi diagram.
        line_thickness (int): Thickness of the Voronoi lines.

    Returns:
        image object (PIL.Image.Image): The transformed image.
    """

    # Convert image to numpy array
    img_array = np.array(image)
    height, width, _ = img_array.shape

    # Generate random seed points
    points = np.random.rand(num_points, 2) * [width, height]

    # Create Voronoi diagram
    vor = Voronoi(points)

    # Create a blank image to draw the diagram
    output_img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(output_img)

    # Draw the actual Voronoi regions
    for region_index in vor.regions:
        if not region_index or -1 in region_index:  # Skip unbounded regions
            continue

        # Get the vertices for specified region
        polygon = [tuple(vor.vertices[i]) for i in region_index if 0 <= i < len(vor.vertices)]
        
        # Skip degenerate polygons
        if len(polygon) < 3:
            continue
        
        