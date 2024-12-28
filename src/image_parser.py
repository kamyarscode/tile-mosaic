from PIL import Image
import numpy as np

def open_image(input_path):
    """
    Opens an image file.
    
    Args:
        input_path (str): The file path to the input image.
    """

    try:
        # Open the image
        image = Image.open(input_path)
        print(f"Image successfully opened: {input_path}")
        
    
    except Exception as e:
        print(f"Error opening image: {e}")

    return image

def export_image(image, output_path):
    """
    Exports image file to path.
    
    Args:
        image (obj): The image object to be exported.
        output_path (str): The file path to save the exported image.
    """
    
    try:    
        # Save the image to the output path
        image.save(output_path)
        print(f"Image successfully saved to: {output_path}")

    except Exception as e:
        print(f"Error exporting image: {e}")


def rgb_to_color_name(rgb: tuple):
    """
    Converts an RGB tuple to a human-readable color name using basic predefined colors.

    Args:
        rgb (tuple): A tuple of (R, G, B) values.

    Returns:
        str: A string name for the color.
    """
    # Predefined color dictionary
    color_names = {
        (255, 0, 0): "Red",
        (0, 255, 0): "Green",
        (0, 0, 255): "Blue",
        (255, 255, 0): "Yellow",
        (0, 255, 255): "Cyan",
        (255, 0, 255): "Magenta",
        (0, 0, 0): "Black",
        (255, 255, 255): "White",
        (128, 128, 128): "Gray"
    }

    # Find the closest color by Euclidean distance
    closest_color = None
    min_distance = float("inf")
    for color, name in color_names.items():
        distance = np.sqrt(sum((component1 - component2) ** 2 for component1, component2 in zip(rgb, color)))
        if distance < min_distance:
            min_distance = distance
            closest_color = name

    return closest_color or "Unknown"

