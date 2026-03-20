import cv2
import pytesseract
import numpy as np


# ------------------------------------------------
# LOAD IMAGE
# ------------------------------------------------
image = cv2.imread("testimg2.jpg")

if image is None:
    raise ValueError("Image not found. Check file path.")

# ------------------------------------------------
# MINIMAL PREPROCESSING (REAL-TIME SAFE)
# ------------------------------------------------
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gray = cv2.resize(
    gray,
    None,
    fx=1.5,
    fy=1.5,
    interpolation=cv2.INTER_CUBIC
)


# 2️⃣ pic(1) CLAHE – local contrast enhancement
clahe = cv2.createCLAHE(
    clipLimit=2,      # Increase contrast without blowing highlights
    tileGridSize=(8,8)
)
contrast = clahe.apply(gray)

gamma = 0.7
look_up = np.array([
    ((i / 255.0) ** (1 / gamma)) * 255
    for i in np.arange(256)
]).astype("uint8")

enhanced = cv2.LUT(contrast, look_up)


cv2.imwrite("preprocessed_image.jpg", enhanced)  # Save preprocessed image for reference

# ------------------------------------------------
# TESSERACT CONFIGURATION (GENERALIZED)
# ------------------------------------------------

# ------------------------------------------------
# OCR EXTRACTION
# ------------------------------------------------
text = pytesseract.image_to_string(
    enhanced,
    config="--oem 3 --psm 11 --dpi 300 ",
    lang="eng"
)


# ------------------------------------------------
# LIGHT OUTPUT CLEANING (STRUCTURE ONLY)
# ------------------------------------------------
clean_text = "\n".join(
    line.strip() for line in text.splitlines() if line.strip()
)

# ------------------------------------------------
# OUTPUT
# ------------------------------------------------

with open("ocr_output.txt", "w", encoding="utf-8") as f:
    f.write(clean_text)

print("OCR text saved to ocr_output.txt")

# =================================================
# AI-ASSISTED LANGUAGE CORRECTION (POST-OCR)
# =================================================

from difflib import get_close_matches

# ------------------------------------------------
# LOAD SIMPLE ENGLISH VOCABULARY
# ------------------------------------------------
# This is lightweight and real-time safe
vocabulary = set()

try:
    with open("/usr/share/dict/words", "r", errors="ignore") as f:
        for w in f:
            vocabulary.add(w.strip().lower())
except FileNotFoundError:
    # Fallback minimal vocabulary (safe)
    vocabulary = {
        "the","and","of","to","in","for","with","on","this","that",
        "project","system","text","language","translation","smart",
        "glasses","real","time","digital","printed","audio","output",
        "user","camera","design","development","reading","foreign"
    }

# ------------------------------------------------
# SIMPLE AI-STYLE WORD CORRECTION
# ------------------------------------------------
def language_correct_word(word):
    # Do not touch short words, numbers, or symbols
    if len(word) < 3 or not word.isalpha():
        return word

    if word.lower() in vocabulary:
        return word

    matches = get_close_matches(
        word.lower(),
        vocabulary,
        n=1,
        cutoff=0.8
    )

    if matches:
        # Preserve original capitalization
        return matches[0].capitalize() if word[0].isupper() else matches[0]

    return word

# ------------------------------------------------
# APPLY CORRECTION TO OCR TEXT
# ------------------------------------------------
corrected_lines = []

for line in clean_text.splitlines():
    words = line.split()
    corrected_words = [language_correct_word(w) for w in words]
    corrected_lines.append(" ".join(corrected_words))

final_text = "\n".join(corrected_lines)

# ------------------------------------------------
# SAVE AI-ASSISTED OCR OUTPUT
# ------------------------------------------------
with open("ocr_op1.txt", "w", encoding="utf-8") as f:
    f.write(final_text)

print("ocr_op1.txt")


