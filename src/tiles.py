
import numpy as np
from scipy.spatial import Voronoi
from PIL import Image, ImageDraw
from collections import defaultdict

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

# We need to create a mask so we ignore irrelevant areas of the images. This includes white or black spaces in corners of the image.
def create_mask(image, threshold=240):
    """
    Creates a binary mask to identify relevant areas in an image.
    White or irrelevant regions are excluded based on a threshold. 
    Adjust threshold value until optimal is achieved.

    Args:
        image (PIL.Image.Image): Input pillow image object
        threshold (int): Intensity value to classify regions as irrelevant (default 240).

    Returns:
        numpy.ndarray: Resultant binary mask (1 for relevant areas, 0 for irrelevant areas).
    """

    # Convert image to grayscale
    img_array = np.array(image.convert("L"))  # Convert to grayscale (L mode)

    # Create a binary mask using thresholding
    mask = img_array < threshold  # Relevant areas are below the threshold

    return mask.astype(np.uint8)  # Convert to 0 and 1 values

# In order to decrease amount of voronoi regions, we need to merge regions with similar colors. Doing this will make it easier
# for later when the images are printed and used.
def merge_similar_regions(vor, region_colors, color_threshold):
    """
    Merges adjacent Voronoi regions with similar colors. If it adheres to specified threshold, they should merge and output avg color.

    Args:
        vor (scipy.spatial.Voronoi): Object for Voronoi diagram.
        region_colors (dict): Dict mapping region index to the average color.
        color_threshold (float): Value threshold for what colors to merge.

    Returns:
        dict: Merged regions as a mapping of region indices to new polygons.
    """

    # Function to calculate color difference between 2 colors.
    def color_distance(color1, color2):
        return np.linalg.norm(np.array(color1) - np.array(color2))

    # Build adjacency list for regions
    region_adjacency = defaultdict(set)
    
    for point_indices in vor.ridge_points:
        region1, region2 = vor.point_region[point_indices]
        if region1 != -1 and region2 != -1 and region1 in region_colors and region2 in region_colors:
            # Add to region adjacent
            region_adjacency[region1].add(region2)
            region_adjacency[region2].add(region1)

    # Merge regions based on color similarity
    visited = set()
    merged_regions = {}

    for region_index, color in region_colors.items():
        if region_index in visited:
            continue

        # Initialize a new merged region
        merged_region = {region_index}
        merged_polygon = [tuple(vor.vertices[i]) for i in vor.regions[region_index] if 0 <= i < len(vor.vertices)]
        merged_color = color
        region_queue = [region_index]
        visited.add(region_index)

        while region_queue:
            current_region = region_queue.pop()
            for neighbor in region_adjacency[current_region]:
                if neighbor in visited:
                    continue

                neighbor_color = region_colors[neighbor]
                if color_distance(merged_color, neighbor_color) < color_threshold:
                    # Merge the neighbor into the region
                    visited.add(neighbor)
                    merged_region.add(neighbor)
                    region_queue.append(neighbor)

                    # Update the polygon which are a collection of vertices
                    neighbor_polygon = [tuple(vor.vertices[i]) for i in vor.regions[neighbor] if 0 <= i < len(vor.vertices)]
                    merged_polygon.extend(neighbor_polygon)

        # Calculate the final average color for the merged region
        merged_color = np.mean([region_colors[region] for region in merged_region], axis=0)
        merged_regions[region_index] = (merged_polygon, tuple(map(int, merged_color)))

    return merged_regions

def voronoi_with_region_merging(image, num_points=500, line_thickness=1, color_threshold=30):
    """
    Applies a Voronoi transformation with region merging based on color similarity.

    Args:
        image (PIL.Image.Image): The input image as a Pillow object.
        num_points (int): Number of seed points for the Voronoi.
        line_thickness (int): Thickness of the lines.
        color_threshold (float): Allowable color difference between regions.

    Returns:
        PIL.Image.Image: The transformed image as a Pillow object.
    """
    
    # Convert image to numpy array
    img_array = np.array(image)
    height, width, _ = img_array.shape

    # Random seed points
    points = [(np.random.randint(0, width), np.random.randint(0, height)) for _ in range(num_points)]

    # Create Voronoi
    vor = Voronoi(points)

    # Compute average colors for all regions
    region_colors = {}
    for region_index in range(len(vor.regions)):
        if -1 in vor.regions[region_index]:  # Skip unbounded regions
            continue

        # Get the region's vertices
        polygon = [tuple(vor.vertices[i]) for i in vor.regions[region_index] if 0 <= i < len(vor.vertices)]
        if len(polygon) < 3:
            continue  

        # Calculate average color of the region
        temp_mask = np.zeros(img_array.shape[:2], dtype=bool)
        x, y = zip(*polygon)
        rr, cc = skpolygon(np.array(y), np.array(x), shape=img_array.shape[:2])
        temp_mask[rr, cc] = True
        region_pixels = img_array[temp_mask]
        avg_color = tuple(region_pixels.mean(axis=0).astype(int)) if len(region_pixels) > 0 else (255, 255, 255)

        region_colors[region_index] = avg_color

    # Merge similar regions
    merged_regions = merge_similar_regions(vor, region_colors, color_threshold)

    # Draw the merged regions
    output_img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(output_img)

    for _, (polygon, color) in merged_regions.items():
        polygon = [(int(x), int(y)) for x, y in polygon]
        draw.polygon(polygon, fill=color)

        if line_thickness > 0:
            draw.line(polygon + [polygon[0]], fill=(0, 0, 0), width=line_thickness)

    return output_img

# Brief test here
# Load image
image_path = "img_src/car.jpg"
input_image = open_image(image_path).convert("RGB")

# Apply Voronoi mosaic with region merging
output_image = voronoi_with_region_merging(input_image, num_points=1000, line_thickness=1, color_threshold=60)

# Save or display the output
output_image.show()



        