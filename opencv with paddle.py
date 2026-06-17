# Install pip install paddleocr
# then cpu version do install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/ 

import os
import cv2
import numpy as np
from paddleocr import PaddleOCR

def preprocess_glasses_frame(image_path):
    """
    Applies real-time safe OpenCV transformations optimized for 
    wearable smart glasses feeds under uneven lighting.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found at {image_path}. Check file path.")

    # 1. Convert to Grayscale (Drops channels to save CPU overhead)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. 2x Upscale using Cubic Interpolation for fine text feature reconstruction
    gray = cv2.resize(
        gray,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_CUBIC
    )

    # 3. Apply CLAHE to neutralize environmental variations and local shadows
    clahe = cv2.createCLAHE(
        clipLimit=2,
        tileGridSize=(8, 8)
    )
    contrast = clahe.apply(gray)

    # 4. Fast execution Gamma Correction Curve (Lifts low-light details via 256-byte mapping)
    gamma = 0.7
    look_up = np.array([
        ((i / 255.0) ** gamma) * 255
        for i in np.arange(256)
    ]).astype("uint8")

    enhanced = cv2.LUT(contrast, look_up)

    # Save the optimized matrix to a temporary file for the backend pipeline to parse
    temp_output_path = "preprocessed_image.jpg"
    cv2.imwrite(temp_output_path, enhanced)
    
    return temp_output_path


# =====================================================================
# CORE SYSTEM EXECUTION PIPELINE
# =====================================================================
if __name__ == '__main__':

    # 1. Initialize the official High-Level PP-OCRv6 Pipeline
    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        text_detection_model_name="PP-OCRv6_medium_det",  # High-precision bounding boxes for small fonts
        text_recognition_model_name="PP-OCRv6_medium_rec", # Deep token parsing for unclear characters
        use_doc_unwarping=False,
        use_textline_orientation=False,
        engine="paddle",
        enable_mkldnn=False  # <--- THIS IS THE CRITICAL BUG FIX FOR WINDOWS CPU RUNTIMES
    )

    # 2. Assign target image file pathway
    img_path = "testimg2.jpg"  # Simulate a snapshot captured by the smart glasses

    if not os.path.exists(img_path):
        print(f"\n[ERROR] Could not find file at: {img_path}")
        print("-> Please ensure your test image is dropped into your working directory workspace.")
    else:
        print("\n[INFO] Cleaning up raw glasses camera frame via OpenCV...")
        # Run the image through your custom LUT/CLAHE preprocessing pipeline
        optimized_path = preprocess_glasses_frame(img_path)
        
        print("[INFO] Running high-accuracy PP-OCRv6 inference...")
        # Feed the optimized image path directly to the prediction pipeline
        result = ocr.predict(optimized_path)
        
        # --- PARAGRAPH STORAGE LIST ---
        paragraph_lines = []
        
        # 3. Loop through the output container object, save data, and extract text strings
        for res in result:
            res.save_to_img("output")  # Saves visual verification bounding boxes to workspace
            res.save_to_json("output")  # Saves structured JSON data matrix
            
            # Look directly inside the res dictionary keys for 'rec_texts'
            if 'rec_texts' in res:
                for text in res['rec_texts']:
                    paragraph_lines.append(text.strip())

        # 4. Join all extracted text blocks together into a single paragraph string
        full_paragraph = " ".join(paragraph_lines)

        print("\n" + "="*60)
        print("                 FINAL EXTRACTED TEXT                          ")
        print("="*60)
        print(full_paragraph)
        print("="*60)
        print("[SUCCESS] Check the 'output' directory folder on your left panel to see files!")

        # 5. Clean up the temporary file safely to preserve system storage space
        if os.path.exists(optimized_path):
            os.remove(optimized_path)