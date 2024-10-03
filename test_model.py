import cv2
import pickle

from utils import get_face_landmarks

emotions = ['HAPPY', 'SAD', 'SURPRISE']

# Load the trained model
with open('./model', 'rb') as f:
    model = pickle.load(f)


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Exit if the frame isn't captured properly

    face_landmarks = get_face_landmarks(frame, draw=True, static_image_mode=False)

    output  = model.predict([face_landmarks])


    # Enhance label appearance with background rectangle and better font size
    label = emotions[int(output[0])]
    text_position = (50, 50)
    font_scale = 2
    font_color = (255, 255, 255)
    background_color = (255, 0, 0)

    # Calculate text size
    text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
    text_w, text_h = text_size

    # Add background rectangle for better visibility
    cv2.rectangle(frame, (text_position[0] - 10, text_position[1] - text_h - 10),
                  (text_position[0] + text_w + 10, text_position[1] + 10), background_color, -1)

    # Add the label text on top of the rectangle
    cv2.putText(frame, label, text_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(25) != -1:
        break

cap.release()
cv2.destroyAllWindows()
