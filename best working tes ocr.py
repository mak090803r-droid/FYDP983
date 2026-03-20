import cv2
import pytesseract
import time
import numpy as np
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
    fx=2,
    fy=2,
    interpolation=cv2.INTER_CUBIC
)

clahe = cv2.createCLAHE(
    clipLimit=2,
    tileGridSize=(8, 8)
)
contrast = clahe.apply(gray)

gamma = 0.7
look_up = np.array([
    ((i / 255.0) ** (1 / gamma)) * 255
    for i in np.arange(256)
]).astype("uint8")

enhanced = cv2.LUT(contrast, look_up)

cv2.imwrite("preprocessed_image.jpg", enhanced)

# ------------------------------------------------
# TESSERACT OCR
# ------------------------------------------------
text = pytesseract.image_to_string(
    enhanced,
    config="--oem 3 --psm 6 --dpi 300",
    lang="eng"
)

clean_text = "\n".join(
    line.strip() for line in text.splitlines() if line.strip()
)

with open("ocr_output.txt", "w", encoding="utf-8") as f:
    f.write(clean_text)

print("OCR text saved to ocr_output.txt")

end = time.time()
print("Execution time:", end - start, "seconds")