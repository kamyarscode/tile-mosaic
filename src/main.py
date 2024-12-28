
from image_parser import open_image
from tiles import voronoi_with_region_merging
from image_parser import rgb_to_color_name
def test():
    
    # Brief test here
    # Load image
    image_path = "img_src/car.jpg"
    input_image = open_image(image_path).convert("RGB")

    # Apply Voronoi mosaic with region merging
    output_image = voronoi_with_region_merging(input_image, num_points=1000, line_thickness=1, color_threshold=60)

    # Save or display the output
    output_image.show()

def main():
    print (rgb_to_color_name((63,84,246)))

if __name__ == "__main__":
    main()

