import os
import time
from paddleocr import PaddleOCR

def test_ocr_version(version_name, image_path):
    print(f"\n[+] Initializing {version_name} Engine...")
    
    # Initialize PaddleOCR engine targeting specific version weights
    start_init = time.time()
    ocr = PaddleOCR(
        use_angle_cls=True, 
        lang='en', 
        use_gpu=False, 
        ocr_version=version_name,
        show_log=False
    )
    init_time = time.time() - start_init
    print(f"    Loaded in: {init_time:.2f} seconds")

    # Measure actual text processing speed
    print(f"[+] Processing '{image_path}' using {version_name}...")
    start_inference = time.time()
    result = ocr.ocr(image_path, cls=True)
    inference_time = time.time() - start_inference
    
    print(f"\n--- {version_name} Results (Inference Time: {inference_time:.3f}s) ---")
    if not result or result[0] is None:
        print("    No text segments detected.")
        return inference_time

    # Parse and print text detections and confidence flags
    for page in result:
        for line in page:
            text = line[1][0]
            confidence = line[1][1]
            print(f"    Text: '{text}' | Confidence: {confidence:.2f}")
            
    return inference_time

if __name__ == '__main__':
    # Define target image
    test_image = 'testimg2.jpg'
    
    if not os.path.exists(test_image):
        print(f"[-] Error: Please drop an image named '{test_image}' inside this directory first.")
    else:
        print("==================================================")
        # 1. Run Legacy PP-OCRv4 Baseline
        v4_time = test_ocr_version('PP-OCRv4', test_image)
        
        print("==================================================")
        # 2. Run New PP-OCRv6 Pipeline
        v6_time = test_ocr_version('PP-OCRv6', test_image)
        print("==================================================")
        
        # Summary speed metrics 
        print("\n=== PERFORMANCE COMPARISON ===")
        print(f"PP-OCRv4 Inference Time: {v4_time:.3f} seconds")
        print(f"PP-OCRv6 Inference Time: {v6_time:.3f} seconds")
        if v6_time < v4_time:
            speedup = v4_time / v6_time
            print(f"⚡ PP-OCRv6 is running ~{speedup:.1f}x faster on your CPU!")