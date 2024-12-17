
from image_parser import open_image
from tiles import voronoi_with_region_merging
def main():
    
    # Brief test here
    # Load image
    image_path = "img_src/car.jpg"
    input_image = open_image(image_path).convert("RGB")

    # Apply Voronoi mosaic with region merging
    output_image = voronoi_with_region_merging(input_image, num_points=1000, line_thickness=1, color_threshold=60)

    # Save or display the output
    output_image.show()

if __name__ == "__main__":
    main()
