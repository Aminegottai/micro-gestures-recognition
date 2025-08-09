import numpy as np
import matplotlib.pyplot as plt

keypoints = np.load("dataset/keypoints/pointer_001.npy")
for frame in keypoints[:10]:
    x = frame[::2]
    y = frame[1::2]
    plt.scatter(x, y)
    plt.title("Points clés d'une frame")
    plt.gca().invert_yaxis()
    plt.show()