from PIL import Image

def open_image(input_path, output_path):
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

