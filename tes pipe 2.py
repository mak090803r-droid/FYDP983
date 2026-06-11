import cv2
import pytesseract
import re
from difflib import SequenceMatcher

cap = cv2.VideoCapture("/run/media/black/New Volume/Sem 8/FYDP/Project Main/Code/inputs/VID_20260417_220740.mp4")

frame_count = 0
all_text = []

def is_similar(a, b, threshold=0.75):
    return SequenceMatcher(None, a, b).ratio() > threshold

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % 30 != 0:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2)

    _, gray = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)

    filtered_words = []
    for i in range(len(data['text'])):
        try:
            conf = int(data['conf'][i])
        except:
            conf = -1

        if conf > 60 and data['text'][i].strip() != "":
            filtered_words.append(data['text'][i])

    text = " ".join(filtered_words)

    text = re.sub(r'[^a-zA-Z0-9.,!?\'"()\- ]+', '', text)

    # ✅ THIS MUST BE INSIDE LOOP
    if any(is_similar(text, t) for t in all_text):
        continue

    all_text.append(text)

    print(f"\nFrame {frame_count}:")
    print(text)

cap.release()

with open("final_ocr_output.txt", "w", encoding="utf-8") as f:
    f.write("\n\n".join(all_text))

print("\n✅ OCR complete")