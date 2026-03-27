import os
os.environ.setdefault('HUB_DATASET_ENDPOINT', 'https://modelscope.cn/api/v1/datasets')

from paddleocr import PaddleOCR


# Set image path
image_path = "testimg2.jpg"

# Initialize OCR (CPU-only, English, no angle detection)
ocr = PaddleOCR(use_textline_orientation=False, lang='en')  # paddleocr>=2.6.1

# Run OCR
results = ocr.ocr(image_path)

# Extract text lines
lines = []
for line in results[0]:
    text, confidence = line[1]
    lines.append(text)

# Save to file
with open("paddle1.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print("✅ OCR output saved to paddle1.txt")