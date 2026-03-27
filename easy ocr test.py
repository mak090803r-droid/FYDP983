import easyocr
import numpy as np
import cv2

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

reader = easyocr.Reader(['en'], gpu=False)  # Use CPU only
results = reader.readtext('preprocessed_image.jpg')

# Extract text only (ignore boxes and confidence)
lines = [text for _, text, _ in results]

# Save to file
with open("easyocr_output.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print("OCR output saved to easyocr_output.txt")