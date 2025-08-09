import numpy as np
import os
import pandas as pd

def augment_keypoints(keypoints, rotation=10, scale=0.1, noise=0.01):
    theta = np.radians(np.random.uniform(-rotation, rotation))
    rot_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    keypoints = keypoints.reshape(-1, 2) @ rot_matrix
    keypoints *= np.random.uniform(1 - scale, 1 + scale)
    keypoints += np.random.normal(0, noise, keypoints.shape)
    return keypoints.reshape(-1)

keypoints_dir = "dataset/keypoints"
output_dir = "dataset/keypoints_augmented"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

labels_df = pd.read_csv("dataset/labels.csv")
augmented_labels = []

for filename in os.listdir(keypoints_dir):
    if filename.endswith(".npy"):
        keypoints = np.load(os.path.join(keypoints_dir, filename))
        video_name = filename.replace(".npy", ".mp4")
        intention = labels_df[labels_df['video_file'] == video_name]['intention'].iloc[0]
        
        for i in range(3):
            augmented = np.array([augment_keypoints(frame) for frame in keypoints])
            aug_filename = f"{filename[:-4]}_aug_{i}.npy"
            np.save(os.path.join(output_dir, aug_filename), augmented)
            augmented_labels.append({"video_file": aug_filename.replace(".npy", ".mp4"), 
                                    "intention": intention})
        print(f"Augmenté {filename}")

augmented_df = pd.DataFrame(augmented_labels)
updated_labels = pd.concat([labels_df, augmented_df], ignore_index=True)
updated_labels.to_csv("dataset/labels.csv", index=False)
print("Fichier labels.csv mis à jour avec les données augmentées")