import os
import cv2

def batch_rotate_images(input_folder: str, output_folder: str):
    """
    Reads all images in the input_folder, rotates them 90 degrees clockwise,
    and saves them to the output_folder.
    """
    # Create the output folder if it doesn't already exist
    os.makedirs(output_folder, exist_ok=True)

    valid_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
    processed_count = 0

    print(f"Scanning folder: {input_folder}...")

    for file_name in os.listdir(input_folder):
        if file_name.lower().endswith(valid_extensions):
            input_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, file_name)

            # Read the image using OpenCV
            img = cv2.imread(input_path)

            if img is not None:
                # Rotate the image 90 degrees clockwise
                # Note: If they need to go the other way, change this to cv2.ROTATE_90_COUNTERCLOCKWISE
                rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

                # Save the rotated image to the new folder
                cv2.imwrite(output_path, rotated_img)
                processed_count += 1
                print(f"✅ Rotated and saved: {file_name}")
            else:
                print(f"⚠️ Warning: Could not read {file_name}")

    print(f"\n🎉 Successfully rotated {processed_count} images and saved them to {output_folder}!")

if __name__ == "__main__":
    # ⚠️ EDIT THESE PATHS TO MATCH YOUR FOLDERS
    
    # The folder where your current sideways images are saved
    INPUT_DIR = "data/liveness_dataset/spoof" 
    
    # The folder where you want the perfectly rotated images to be saved
    OUTPUT_DIR = "data/liveness_dataset/spoof_rotated" 
    
    batch_rotate_images(INPUT_DIR, OUTPUT_DIR)