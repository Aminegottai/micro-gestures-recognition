import cv2
import os

output_dir = "dataset/videos"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

gesture_name = input("Entrez le nom de l'intention (ex. : pointer, saisir, annuler) : ")
video_count = len([f for f in os.listdir(output_dir) if gesture_name in f]) + 1
cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
recording = False

print("Appuyez sur 's' pour commencer, 'e' pour arrêter, 'q' pour quitter")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if recording:
        cv2.putText(frame, 'RECORDING', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('Enregistrement', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s') and not recording:
        recording = True
        video_path = f"{output_dir}/{gesture_name}_{video_count:03d}.mp4"
        out = cv2.VideoWriter(video_path, fourcc, 30, (640, 480))
        print(f"Enregistrement de {video_path}")
    elif key == ord('e') and recording:
        recording = False
        video_count += 1
        out.release()
        print(f"Vidéo {video_path} sauvegardée")
    elif key == ord('q'):
        break
    if recording:
        out.write(frame)

cap.release()
cv2.destroyAllWindows()