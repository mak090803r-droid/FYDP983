import cv2
import pytesseract
import re

# Input video
cap = cv2.VideoCapture("/run/media/black/New Volume/Sem 8/FYDP/Project Main/Code/inputs/VID_20260417_220740.mp4")  # <-- put your video path here

frame_count = 0
all_text = []
last_text = ""

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # 🔥 Skip frames (process every 20th frame)
    if frame_count % 20 != 0:
        continue

    # 🧠 Preprocessing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2)

    # Threshold (helps a lot for printed text)
    _, gray = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    # 🎯 OCR with confidence filtering
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

    # 🧼 Clean text
    text = re.sub(r'[^a-zA-Z0-9.,!?\'"()\- ]+', '', text)

    # 🔁 Remove duplicates
    if text.strip() == "" or text == last_text:
        continue

    last_text = text
    all_text.append(text)

    print(f"\nFrame {frame_count}:")
    print(text)

cap.release()

# 💾 Save output
with open("final_ocr_output.txt", "w", encoding="utf-8") as f:
    f.write("\n\n".join(all_text))

print("\n✅ OCR complete. Saved to final_ocr_output.txt")


