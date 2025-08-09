import os
import pandas as pd

videos_dir = "dataset/videos"
labels = []

for video_file in os.listdir(videos_dir):
    if video_file.endswith(".mp4"):
        intention = video_file.split("_")[0].lower()
        labels.append({"video_file": video_file, "intention": intention})

df = pd.DataFrame(labels)
df.to_csv("dataset/labels.csv", index=False)
print("Fichier labels.csv mis à jour")