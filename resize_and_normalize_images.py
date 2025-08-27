import os
from PIL import Image
import numpy as np

# Define constants
IMAGE_DIRS = ["For Lee/pageOne/", "For Lee/pageTwo/", "For Lee/pageThree/"]
OUTPUT_DIR = "Processed_Images/"
IMAGE_SIZE = (224, 224)

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Function to normalize pixel values
def normalize_image(image):
    image_array = np.array(image).astype(np.float32) / 255.0  # Normalize to [0, 1]
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalized_image = (image_array - mean) / std
    return Image.fromarray((normalized_image * 255).astype(np.uint8))

# Process images
for image_dir in IMAGE_DIRS:
    for filename in os.listdir(image_dir):
        if filename.endswith(".png"):
            input_path = os.path.join(image_dir, filename)
            output_path = os.path.join(OUTPUT_DIR, filename)

            try:
                # Open and resize the image
                with Image.open(input_path) as img:
                    img_resized = img.resize(IMAGE_SIZE)

                    # Normalize the image
                    img_normalized = normalize_image(img_resized)

                    # Save the processed image
                    img_normalized.save(output_path)
                    print(f"Processed and saved: {output_path}")
            except Exception as e:
                print(f"Error processing {input_path}: {e}")