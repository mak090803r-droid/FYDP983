import os
import cv2
from paddleocr import PaddleOCR
import time

start=time.time()


# ------------------------------------------------

def preprocess_glasses_frame(image_path):
    # Load raw image
    img = cv2.imread(image_path)
    if img is None:
        return image_path  # Fallback to original if file reading fails
        
    # 1. Convert to Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Smart Resize (Normalize width to 1600px to scale up small fonts safely)
    target_width = 1600.0
    h, w = gray.shape[:2]
    if w < target_width:
        scale = target_width / w
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    elif w > 2400.0: # Downscale massive images to save CPU processing time
        scale = 2000.0 / w
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    # 3. Flatten Uneven Lighting using CLAHE
    # clipLimit controls contrast strength; tileGridSize splits the image into local cells
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    
    # 4. Remove Sensor Noise while preserving crisp text edges
    # d=9 is the pixel neighborhood; higher sigma values smooth larger areas of noise
    denoised = cv2.bilateralFilter(equalized, d=9, sigmaColor=75, sigmaSpace=75)
    
    # Save the optimized matrix to a temporary file
    temp_processed_path = "temp_glasses_input.jpg"
    cv2.imwrite(temp_processed_path, denoised)
    return temp_processed_path

# =====================================================================
# SYSTEM EXECUTION BLOCK
# =====================================================================

ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    text_detection_model_name="PP-OCRv6_medium_det",  # Best for blurry/small fonts
    text_recognition_model_name="PP-OCRv6_medium_rec",
    use_doc_unwarping=False,
    use_textline_orientation=False,
    engine="paddle",
    enable_mkldnn=False 
)

img_path = "testimg2.jpg"  # Simulate a snapshot captured by the smart glasses

if not os.path.exists(img_path):
    print(f"[ERROR] Could not find file at: {img_path}")
else:
    print("[INFO] Cleaning up raw glasses camera frame...")
    # Run the image through the custom preprocessing pipeline
    optimized_path = preprocess_glasses_frame(img_path)
    
    print("[INFO] Running PP-OCRv6 inference...")
    result = ocr.predict(optimized_path)
    
    paragraph_lines = []
    for res in result:
        res.save_to_img("output")
        res.save_to_json("output")
        if 'rec_texts' in res:
            for text in res['rec_texts']:
                paragraph_lines.append(text.strip())

    print("\n" + "="*60)
    print("                 FINAL EXTRACTED TEXT                          ")
    print("="*60)
    print(" ".join(paragraph_lines))
    print("="*60)

    # Clean up the temporary file safely
    if os.path.exists(optimized_path):
        os.remove(optimized_path)

        end = time.time()
print("Execution time:", end - start, "seconds")