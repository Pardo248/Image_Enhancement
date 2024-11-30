import os
import numpy as np
import cv2

# Function to load and process the image
def load_image(image_path):
    # Load image using OpenCV
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = img / 255.0  # Normalize pixel values between 0 and 1
    return img

# Function to downscale the image resolution
def downscale_image(img, scale=0.7): 
    h, w, _ = img.shape
    low_res_img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    return low_res_img#seeee

# Function to save an image to a file
def save_image(img, output_path):
    img = (img * 255).astype(np.uint8)  # Denormalize pixel values (back to 0-255 range)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert back to BGR for saving
    cv2.imwrite(output_path, img)

# Main function
def main():
    input_folder = r"buena_calidad"  # Folder containing images
    output_folder = r"mala_calidad"#os.path.join(input_folder, "resized_images")  # Folder to save resized images
    os.makedirs(output_folder, exist_ok=True)  # Create output folder if it doesn't exist

    scale = 0.3  # Scale factor for resizing
    
    # Iterate over all image files in the folder
    for filename in os.listdir(input_folder):
        # Construct full file path
        input_path = os.path.join(input_folder, filename)
        
        # Check if it's a file and an image
        if os.path.isfile(input_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                # Process the image
                img = load_image(input_path)
                low_res_img = downscale_image(img, scale=scale)
                
                # Save the resized image
                output_path = os.path.join(output_folder, f"low_res_{filename}")
                save_image(low_res_img, output_path)
                #print(f"Processed and saved: {output_path}")
            except Exception as e:
                print(f"Failed to process {input_path}: {e}")

    print("All images processed!")

if __name__ == "__main__":
    main()
