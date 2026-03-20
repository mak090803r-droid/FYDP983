import cv2
import pytesseract
import time
start = time.time()

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
    fx=2.5,
    fy=2.5,
    interpolation=cv2.INTER_CUBIC
)
cv2.imwrite("preprocessed_image.jpg", gray)  # Save preprocessed image for reference

# ------------------------------------------------
# TESSERACT CONFIGURATION (GENERALIZED)
# ------------------------------------------------

# ------------------------------------------------
# OCR EXTRACTION
# ------------------------------------------------
text = pytesseract.image_to_string(
    gray,
    config="--oem 3 --psm 11 --dpi 300",
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

end = time.time()
print("Execution time:", end - start, "seconds")
