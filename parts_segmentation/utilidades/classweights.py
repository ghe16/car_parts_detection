import os
import numpy as np
from PIL import Image

def calculate_class_weights(mask_dir, num_classes):
    class_counts = np.zeros(num_classes)

    # Iterate over all masks in the dataset
    for mask_name in os.listdir(mask_dir):
        mask_path = os.path.join(mask_dir, mask_name)
        mask = np.array(Image.open(mask_path).convert('L'))  # Load mask as a grayscale image

        # Count the pixels for each class in the mask
        for cls in range(num_classes):
            class_counts[cls] += np.sum(mask == cls)

    # Calculate the total number of pixels
    total_pixels = np.sum(class_counts)

    # Calculate the class weights as the inverse of the class frequencies
    class_weights = total_pixels / (num_classes * class_counts)
    
    return class_weights