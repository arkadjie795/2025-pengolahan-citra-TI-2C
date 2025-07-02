import cv2
from cvzone.HandTrackingModule import HandDetector
import cvzone
import numpy as np

# 🔍 Coba semua index kamera 0-4
def get_available_camera_index():
    for i in range(5):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            cap.release()
            return i
    return None

# 🔌 Temukan kamera yang bisa digunakan
cam_index = get_available_camera_index()
if cam_index is None:
    print("❌ Tidak ada kamera yang tersedia.")
    exit()

cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
cap.set(3, 1280)  # Width
cap.set(4, 720)   # Height

detector = HandDetector(detectionCon=1)
colorR = (255, 0, 255)

class DragRect():
    def __init__(self, posCenter, size=[200, 200]):
        self.posCenter = posCenter
        self.size = size

    def update(self, cursor):
        cx, cy = self.posCenter
        w, h = self.size
        if cx - w // 2 < cursor[0] < cx + w // 2 and \
           cy - h // 2 < cursor[1] < cy + h // 2:
            self.posCenter = cursor

# Membuat daftar kotak
rectList = [DragRect([x * 250 + 150, 150]) for x in range(5)]

while True:
    success, img = cap.read()
    if not success or img is None:
        print("⚠️ Gagal ambil frame. Menunggu kamera...")
        # tampilkan layar hitam sebagai fallback
        black = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.imshow("Image", black)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    img = cv2.flip(img, 1)
    img = detector.findHands(img)
    lmList, _ = detector.findPosition(img)

    if lmList:
        l, _, _ = detector.findDistance(8, 12, img, draw=False)
        if l < 30:
            cursor = lmList[8]
            for rect in rectList:
                rect.update(cursor)

    # Membuat layer transparan
    imgNew = np.zeros_like(img, np.uint8)
    for rect in rectList:
        cx, cy = rect.posCenter
        w, h = rect.size
        cv2.rectangle(imgNew, (cx - w // 2, cy - h // 2),
                      (cx + w // 2, cy + h // 2), colorR, cv2.FILLED)
        cvzone.cornerRect(imgNew, (cx - w // 2, cy - h // 2, w, h), 20, rt=0)

    # Gabungkan layer transparan ke frame utama
    out = img.copy()
    alpha = 0.1
    mask = imgNew.astype(bool)
    out[mask] = cv2.addWeighted(img, alpha, imgNew, 1 - alpha, 0)[mask]

    cv2.imshow("Image", out)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
