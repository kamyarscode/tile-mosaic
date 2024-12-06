
import numpy as np
from scipy.spatial import Voronoi
from PIL import Image, ImageDraw
from skimage.draw import polygon as skpolygon

from image_parser import open_image

# Attempt to do this tile using voronoi transformatin method
def voronoi_mosaic(image, num_points=500, line_thickness=1):
    """
    Applies a Voronoi transformation to an image to create a mosaic effect with default 500 points and line thickness of 1.

    Args:
        image object (PIL.Image.Image): The input image as a Pillow object.
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

    # Create the Voronoi diagram
    vor = Voronoi(points)

    # Create a blank image to draw the mosaic
    output_img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(output_img)

    # Process Voronoi regions
    for region_index in vor.regions:
        if not region_index or -1 in region_index:  # Skip invalid regions
            continue

        # Get the region's vertices
        polygon = [tuple(vor.vertices[i]) for i in region_index if 0 <= i < len(vor.vertices)]
        
        if len(polygon) < 3:  # Skip regions that we dont want generated.
            continue

        # Convert polygon to integer coordinates x,y
        polygon = [(int(x), int(y)) for x, y in polygon]

        # Calculate the average color of the region
        avg_color = get_region_color(img_array, polygon)

        # Fill the region with the average color
        draw.polygon(polygon, fill=avg_color)
        
        # Draw the region boundary lines
        if line_thickness > 0:
            draw.line(polygon + [polygon[0]], fill=(0, 0, 0), width=line_thickness)

    return output_img

# Calculate colors needed for the image outputs to render using polygons and RGB.
def get_region_color(img_array, polygon):
    """
    Calculates the average color of a polygonal region on the image.

    Args:
        img_array (numpy.ndarray): The image as a numpy array.
        polygon (tuple array): array of (x, y) vertices of the polygon.

    Returns:
        tuple: The average color as (R, G, B).
    """

    # Create a mask for the region
    mask = np.zeros(img_array.shape[:2], dtype=bool)
    x, y = zip(*polygon)
    rr, cc = skpolygon(np.array(y), np.array(x), shape=img_array.shape[:2])
    mask[rr, cc] = True

    # Calculate the average color within the mask
    region_pixels = img_array[mask]
    average_color = tuple(region_pixels.mean(axis=0).astype(int))

    return average_color

# # Brief test here
# image_path = "img_src/moon.jpg"
# img = open_image(image_path).convert("RGB")

# # Apply Voronoi mosaic
# output_image = voronoi_mosaic(img, num_points=100, line_thickness=5)

# # Save or display the output
# output_image.show()
# output_image.save("mosaic_test.jpg")



        