import os
import argparse
from PIL import Image
import rawpy
from multiprocessing import Pool
import statistics

def get_folder_file_sizes(folder_path):
    file_sizes = []
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            file_sizes.append(os.path.getsize(filepath))
    return file_sizes

def resize_image(input_path, output_folder, max_, quality):
    try:
        filename = os.path.basename(input_path)
        if filename.lower().endswith('.cr2'):
            # Process CR2 files using rawpy
            with rawpy.imread(input_path) as raw:
                img = raw.postprocess()
                img = Image.fromarray(img)
        else:
            # Process other image files
            with Image.open(input_path) as img:
                img = img.copy()

        # Maintain aspect ratio
        max_width = max_height = max_
        img.thumbnail((max_width, max_height))

        # Save the resized image in the output folder
        output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.jpg")
        img.save(output_path, optimize=True, quality=quality)

        print(f"Resized and saved: {output_path}")
    except Exception as e:
        print(f"Error processing {input_path}: {e}")

def resize_images(input_folder, output_folder, max_, quality):
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Get the list of image paths to process
    image_paths = []
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)

        # Skip non-image files
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.cr2')):
            continue

        image_paths.append(input_path)

    # Create a pool of workers and process images in parallel
    with Pool() as pool:
        pool.starmap(resize_image, [(path, output_folder, max_, quality) for path in image_paths])

def calculate_statistics(file_sizes):
    if not file_sizes:
        return None, None  # Avoid division by zero
    mean_size = statistics.mean(file_sizes)
    stdev_size = statistics.stdev(file_sizes) if len(file_sizes) > 1 else 0
    return mean_size, stdev_size

def main():
    parser = argparse.ArgumentParser(description="Resize images from a folder and save them to another folder.")
    parser.add_argument("input_folder", type=str, help="Path to the folder containing the input images.")
    parser.add_argument("output_folder", type=str, help="Path to the folder to save the resized images.")
    parser.add_argument("--max_side", type=int, default=800, help="Maximum width of the resized images. Default is 800.")
    parser.add_argument("--quality", type=int, default=85, help="Quality of the resized images (1-100). Default is 85.")

    args = parser.parse_args()

    # Get file sizes before processing
    input_file_sizes = get_folder_file_sizes(args.input_folder)
    input_mean, input_stdev = calculate_statistics(input_file_sizes)

    print(f"Input folder statistics:")
    print(f"  Mean file size: {input_mean / (1024):.2f} KB")
    print(f"  Standard deviation of file sizes: {input_stdev / (1024):.2f} KB")

    # Resize the images
    resize_images(args.input_folder, args.output_folder, args.max_side, args.quality)
    print("Image resizing complete.")

    # Get file sizes after processing
    output_file_sizes = get_folder_file_sizes(args.output_folder)
    output_mean, output_stdev = calculate_statistics(output_file_sizes)

    print(f"Output folder statistics:")
    print(f"  Mean file size: {output_mean / (1024):.2f} KB")
    print(f"  Standard deviation of file sizes: {output_stdev / (1024):.2f} KB")

if __name__ == "__main__":
    main()
