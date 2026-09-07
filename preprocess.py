import os
import cv2
import numpy as np
from tqdm import tqdm

def preprocess_image(image_path, target_size=(224, 224)):
    # 1. Load image
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # 2. Resize
    img = cv2.resize(img, target_size)
    
    # 3. Convert to grayscale for some processing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 4. Histogram Equalization (CLAHE for better contrast)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    
    # 5. Noise Reduction (Gaussian Blur)
    blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
    
    # Convert back to BGR/RGB if needed for model
    processed_img = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
    
    return processed_img

def main():
    base_dir = "Training"
    output_dir = "Processed_Training"
    classes = ["glioma_tumor", "meningioma_tumor", "pituitary_tumor"]
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for cls in classes:
        cls_path = os.path.join(base_dir, cls)
        out_cls_path = os.path.join(output_dir, cls)
        
        if not os.path.exists(out_cls_path):
            os.makedirs(out_cls_path)
            
        print(f"Processing class: {cls}")
        files = os.listdir(cls_path)
        for f in tqdm(files):
            img_path = os.path.join(cls_path, f)
            processed = preprocess_image(img_path)
            if processed is not None:
                cv2.imwrite(os.path.join(out_cls_path, f), processed)

    # Repeat for Testing
    base_dir = "Testing"
    output_dir = "Processed_Testing"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for cls in classes:
        cls_path = os.path.join(base_dir, cls)
        out_cls_path = os.path.join(output_dir, cls)
        
        if not os.path.exists(out_cls_path):
            os.makedirs(out_cls_path)
            
        print(f"Processing testing class: {cls}")
        files = os.listdir(cls_path)
        for f in tqdm(files):
            img_path = os.path.join(cls_path, f)
            processed = preprocess_image(img_path)
            if processed is not None:
                cv2.imwrite(os.path.join(out_cls_path, f), processed)

if __name__ == "__main__":
    main()
