# scripts/arduino_live.py
import cv2
import numpy as np
import serial
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# adjust to your Arduino COM port (Windows) or /dev/ttyACM0 (Linux)
ARDUINO_PORT = "COM3"
BAUD_RATE = 9600

labels = ['dry_decomposable', 'dry_recyclable', 'dry_non_degradable', 'wet']
model = load_model('models/waste_classifier.h5')

# open serial and camera
ser = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=1)
cap = cv2.VideoCapture(0)

print("Waiting for Arduino messages...")

while True:
    # read line from Arduino
    line = ser.readline().decode('utf-8', errors='ignore').strip()
    if line:
        print("Arduino:", line)

        if line == "WET":
            # wet decided by moisture sensor
            print("Detected: WET (sensor)")
            # show message on camera feed once
            ret, frame = cap.read()
            if ret:
                cv2.putText(frame, "WET (sensor)", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.imshow("Waste Detection", frame)
                cv2.waitKey(1000)
            ser.write(b"MOVE_WET\n")

        elif line == "DRY":
            # dry: use CNN to classify type
            print("Detected: DRY (sensor), running CNN...")
            ret, frame = cap.read()
            if not ret:
                continue

            img = cv2.resize(frame, (128, 128))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            pred = model.predict(img_array, verbose=0)
            probs = pred[0]
            idx = np.argmax(probs)
            cls = labels[idx]
            conf = probs[idx] * 100

            print(f"CNN: {cls} ({conf:.1f}%)")

            text = f"{cls} ({conf:.1f}%)"
            cv2.putText(frame, text, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Waste Detection", frame)
            cv2.waitKey(1000)

            # still send to dry side mechanically
            ser.write(b"MOVE_DRY\n")

    # keep window responsive and allow quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
ser.close()
cv2.destroyAllWindows()
