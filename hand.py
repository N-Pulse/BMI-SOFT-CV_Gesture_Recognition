import mediapipe as mp 
import cv2 
import numpy as np 
import uuid 
import os 

print("MediaPipe and OpenCV are working.")


mp_drawing = mp.solutions.drawing_utils
mp_hand = mp.solutions.hands

# webcam feed
cap = cv2.VideoCapture(0)
with mp_hand.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()

        # BGR to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image = cv2.flip(image, 1)

        # Set flag
        image.flags.writeable = False

        # Detections
        results = hands.process(image)

        # Set flag to true
        image.flags.writeable = True

        # RGB to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Detections
        print(results)

        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image, hand, mp_hand.HAND_CONNECTIONS,)
            lm = results.multi_hand_landmarks[0].landmark[8]  # index fingertip
            print(f"x={lm.x:.3f}, y={lm.y:.3f}, z={lm.z:.3f}")



        cv2.imshow("Hand tracking", image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

    results.multi_hand_landmarks

# Save the captured images