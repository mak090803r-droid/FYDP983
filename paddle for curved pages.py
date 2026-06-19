import os
import cv2
import numpy as np
import time
from paddleocr import PaddleOCR, TextImageUnwarping
import argostranslate.package
import argostranslate.translate

# ══════════════════════════════════════════════
# OPTIMIZED TUNING KNOBS (Single-Channel Focus)
# ══════════════════════════════════════════════
CONFIG = {
    "book_mode":        True, # Toggle True for curved pages, False for normal flat documents
    "translate_mode":   True,  # Toggle True to translate extracted text, False to skip
    "source_lang":      "zh",  # Source language code: "zh" for Chinese, "fr" for French
    "target_lang":      "en",  # Target language code: "en" for English
    "upscale_factor":   1.5,   
    "clahe_clip":       2,   
    "clahe_grid":       8,     
    "gamma":            0.7,   # 1.2 with inv_gamma brightens midtones safely
    "sharpen":          False,  
    "sharpen_strength": 1.0,   
}

# Pre-compute performance configurations ONCE at startup to save mid-frame CPU cycles
INV_GAMMA = 1.0 / CONFIG["gamma"]
GAMMA_LUT = np.array([
    ((i / 255.0) ** INV_GAMMA) * 255
    for i in range(256)
]).astype("uint8")

CLAHE_ENGINE = cv2.createCLAHE(
    clipLimit=CONFIG["clahe_clip"],
    tileGridSize=(CONFIG["clahe_grid"], CONFIG["clahe_grid"])
)

# Pre-build sharpening matrix
S = CONFIG["sharpen_strength"]
SHARPEN_KERNEL = np.array([
    [-S,    -S,   -S],
    [-S, 1+8*S,   -S],
    [-S,    -S,   -S]
])

def preprocess_optimized(img_path, cfg=CONFIG, save_debug=True):
    img = cv2.imread(img_path)
    if img is None:
        print(f"[ERROR] Could not read: {img_path}")
        return None

    # Step 1: Immediately drop to Grayscale (1 Channel instead of 3)
    # Everything below here now runs 3x faster and consumes 1/3 of the RAM
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 2: Denoise on the small grayscale layout
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Step 3: Streamlined CLAHE (No LAB splitting or merging required!)
    contrast = CLAHE_ENGINE.apply(gray)

    # Step 4: Streamlined Gamma LUT mapping on a single channel
    enhanced = cv2.LUT(contrast, GAMMA_LUT)

    # Step 5: Sharpen edge transitions
    if cfg["sharpen"]:
        enhanced = cv2.filter2D(enhanced, -1, SHARPEN_KERNEL)

    # Step 6: Upscale LAST (INTER_LINEAR is faster and smoother for deep learning)
    h, w = enhanced.shape[:2]
    final_img = cv2.resize(
        enhanced,
        (int(w * cfg["upscale_factor"]), int(h * cfg["upscale_factor"])),
        interpolation=cv2.INTER_LINEAR
    )

    if True:
        cv2.imwrite("debug_preprocessed.jpg", final_img)
        print(f"[DEBUG] {w}x{h} → {final_img.shape[1]}x{final_img.shape[0]} (Single Channel)")

    return final_img

# ══════════════════════════════════════════════
# TRANSLATION ENGINE (Argos Translate)
# ══════════════════════════════════════════════
def setup_translation(src_lang, tgt_lang):
    """Downloads and installs the required language pack on first run.
    Subsequent runs use the cached local package instantly."""
    installed = argostranslate.package.get_installed_packages()
    already_installed = any(
        p.from_code == src_lang and p.to_code == tgt_lang for p in installed
    )
    if not already_installed:
        print(f"[INFO] Downloading language pack: {src_lang} → {tgt_lang} (one-time only)...")
        argostranslate.package.update_package_index()
        available = argostranslate.package.get_available_packages()
        pkg = next(
            (p for p in available if p.from_code == src_lang and p.to_code == tgt_lang),
            None
        )
        if pkg is None:
            print(f"[ERROR] No Argos package found for {src_lang} → {tgt_lang}")
            return False
        pkg.install()
        print(f"[INFO] Language pack installed successfully!")
    else:
        print(f"[INFO] Language pack {src_lang} → {tgt_lang} already cached.")
    return True

def translate_text(text, src_lang, tgt_lang):
    """Translates a block of text using locally installed Argos models."""
    return argostranslate.translate.translate(text, src_lang, tgt_lang)

# ══════════════════════════════════════════════
# MAIN RUNTIME
# ══════════════════════════════════════════════
if __name__ == '__main__':
    start = time.time()

    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        text_detection_model_name="PP-OCRv6_medium_det",
        text_recognition_model_name="PP-OCRv6_medium_rec",
        use_doc_unwarping=False,
        use_textline_orientation=False,
        engine="paddle",
        enable_mkldnn=False
    )

    unwarper = None
    if CONFIG["book_mode"]:
        print("[INFO] Book Mode Active: Loading UVDoc Engine...")
        unwarper = TextImageUnwarping(model_name="UVDoc", engine="paddle")

    img_path = "highres3.jpeg"

    if not os.path.exists(img_path):
        print(f"[ERROR] File not found: {img_path}")
    else:
        print("[INFO] Running optimized preprocessing...")
        t1 = time.time()
        
        if CONFIG["book_mode"]:
            unwarp_result = unwarper.predict(img_path, batch_size=1)
            
            # Extract the corrected frame matrix using the direct top-level key
            for res in unwarp_result:
                unwarped_img = res['doctr_img']
            
            temp_path = "temp_unwarped.jpg"
            cv2.imwrite(temp_path, unwarped_img)
            
            processed = preprocess_optimized(temp_path, save_debug=True)
            cv2.imwrite(temp_path, processed)
        else:
            processed = preprocess_optimized(img_path, save_debug=True)
            temp_path = "temp_processed.jpg"
            cv2.imwrite(temp_path, processed)
            
        t2 = time.time()

        print("[INFO] Running OCR...")
        t3 = time.time()
        result = ocr.predict(temp_path)
        t4 = time.time()

        paragraph_lines = []
        for res in result:
            res.save_to_img("output")
            res.save_to_json("output")
            if 'rec_texts' in res:
                for text in res['rec_texts']:
                    if text.strip():
                        paragraph_lines.append(text.strip())

        full_paragraph = " ".join(paragraph_lines)

        print("\n" + "="*60)
        print("              EXTRACTED PARAGRAPH")
        print("="*60)
        print(full_paragraph)
        print("="*60)

        # ── Translation Stage ──
        t5 = time.time()
        if CONFIG["translate_mode"] and full_paragraph.strip():
            src = CONFIG["source_lang"]
            tgt = CONFIG["target_lang"]
            if setup_translation(src, tgt):
                print(f"\n[INFO] Translating {src} → {tgt}...")
                translated = translate_text(full_paragraph, src, tgt)
                print("\n" + "="*60)
                print(f"         TRANSLATED TEXT ({src} → {tgt})")
                print("="*60)
                print(translated)
                print("="*60)
            else:
                print("[WARN] Translation skipped — language pack not available.")
        t6 = time.time()
        
        print(f"\n[TIMING BREAKDOWN]")
        print(f"  New Preprocessing : {t2-t1:.3f}s  <-- Notice the drop!")
        print(f"  OCR inference     : {t4-t3:.3f}s")
        if CONFIG["translate_mode"]:
            print(f"  Translation       : {t6-t5:.3f}s")
        print(f"  Total Execution   : {time.time()-start:.3f}s")

        if os.path.exists(temp_path):
            os.remove(temp_path)