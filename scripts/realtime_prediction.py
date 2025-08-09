import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

model = load_model('micro_gestures_model.keras')
le = LabelEncoder()
le.classes_ = np.load('label_encoder_classes.npy')

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

fixed_length = 90
sequence = []

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        keypoints = []
        for landmark in results.multi_hand_landmarks[0].landmark:
            keypoints.extend([landmark.x, landmark.y])
        sequence.append(keypoints)
        
        if len(sequence) > fixed_length:
            sequence = sequence[-fixed_length:]
        
        if len(sequence) == fixed_length:
            X = np.array([sequence])
            prediction = model.predict(X, verbose=0)
            intention = le.inverse_transform([np.argmax(prediction[0])])[0]
            cv2.putText(frame, f'Intention: {intention}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    cv2.imshow('Prédiction en temps réel', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()