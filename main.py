import cv2
import os
import numpy as np
from pathlib import Path

def images_to_video(image_folder, output_video, fps=30, sort_key=None):
    """
    Convert a sequence of images to a video file.
    
    Parameters:
    -----------
    image_folder : str
        Path to folder containing image frames
    output_video : str
        Path for output video file (e.g., 'output.mp4')
    fps : int
        Frames per second for the output video (default: 30)
    sort_key : function, optional
        Custom sorting function for image filenames
    """
    
    # Get list of image files
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    images = [f for f in os.listdir(image_folder) 
              if Path(f).suffix.lower() in valid_extensions]
    
    if not images:
        print("No valid images found in the folder!")
        return
    
    # Sort images (by default alphabetically, or use custom sort)
    if sort_key:
        images.sort(key=sort_key)
    else:
        images.sort()
    
    # Read first image to get dimensions
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, channels = frame.shape
    
    # Define video codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID', 'MJPG', 'X264'
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    print(f"Creating video from {len(images)} images...")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps}")
    
    # Write each frame to video
    for i, image_file in enumerate(images):
        image_path = os.path.join(image_folder, image_file)
        frame = cv2.imread(image_path)
        
        if frame is None:
            print(f"Warning: Could not read {image_file}, skipping...")
            continue
            
        # Resize frame if dimensions don't match
        if frame.shape[0] != height or frame.shape[1] != width:
            frame = cv2.resize(frame, (width, height))
        
        video.write(frame)
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(images)} frames")
    
    video.release()
    print(f"\nVideo saved successfully: {output_video}")

# Example usage
if __name__ == "__main__":
    # Your specific path
    images_to_video(
    image_folder=r"/Users/anubhi/Desktop/video/Input",  # Fixed path
    output_video=r"/Users/anubhi/Desktop/video/output_video.mp4",
    fps=30
    )

    # Advanced: Custom sorting (e.g., for numbered frames like frame_001.jpg)
    # images_to_video(
    #     image_folder=r"C:\Users\adi12\Desktop\video\Input",
    #     output_video=r"C:\Users\adi12\Desktop\video\output_video.mp4",
    #     fps=24,
    #     sort_key=lambda x: int(x.split('_')[1].split('.')[0])
    # )