import cv2

# Open camera (0 = default webcam)
cap = cv2.VideoCapture(0)



while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow("Camera", frame)

    key = cv2.waitKey(1)

    # Press 's' to save image
    if key == ord('s'):
    
        cv2.imwrite("/run/media/black/New Volume/Sem 7/FYDP/Project Main/Code/captured_text.jpg", frame)

        print("Image saved")
        break

    # Press 'q' to quit
    if key == ord('q'):
        break

cap.release()
image = cv2.imread("captured_text.jpg")

# 1. Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 2. Blur / noise removal
blur = cv2.GaussianBlur(gray, (5,5), 0)

# Adaptive Thresholding (better for uneven lighting)
thresh = cv2.adaptiveThreshold(
    blur, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11,  # block size (odd number)
    2    # constant subtracted from mean
)

# 4. Resize (if needed)
resized = cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

# Save or display
cv2.imshow("Apdative Threshold", thresh)
cv2.imwrite("/run/media/black/New Volume/Sem 7/FYDP/Project Main/Code/adaptive threshold.jpg", thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.destroyAllWindows()





