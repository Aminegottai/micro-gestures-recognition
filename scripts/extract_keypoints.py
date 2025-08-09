import cv2
import mediapipe as mp
import numpy as np
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
dataset_dir = "dataset/videos"
output_dir = "dataset/keypoints"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for video_file in os.listdir(dataset_dir):
    if video_file.endswith(".mp4"):
        cap = cv2.VideoCapture(os.path.join(dataset_dir, video_file))
        keypoints_sequence = []
        
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
                keypoints_sequence.append(keypoints)
            else:
                keypoints_sequence.append([0] * 42)
        
        np.save(os.path.join(output_dir, video_file.replace(".mp4", ".npy")), np.array(keypoints_sequence))
        cap.release()
        print(f"Points clés extraits pour {video_file}")