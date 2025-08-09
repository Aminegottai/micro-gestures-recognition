import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

fixed_length = 90
dropout_rate = 0.6
lstm_dropout = 0.3
epochs = 50
patience = 5
min_delta = 0.005

labels_df = pd.read_csv("dataset/labels.csv")
X, y = [], []

for _, row in labels_df.iterrows():
    keypoints_file = row['video_file'].replace('.mp4', '.npy')
    keypoints_path = f"dataset/keypoints/{keypoints_file}"
    try:
        keypoints = np.load(keypoints_path)
        if len(keypoints) > fixed_length:
            keypoints = keypoints[:fixed_length]
        elif len(keypoints) < fixed_length:
            keypoints = np.pad(keypoints, ((0, fixed_length - len(keypoints)), (0, 0)), 
                              mode='constant', constant_values=0)
        X.append(keypoints)
        y.append(row['intention'])
    except FileNotFoundError:
        print(f"Fichier non trouvé : {keypoints_path}")
        continue

try:
    X = np.array(X)
    y = np.array(y)
    print(f"Dataset chargé : {X.shape} échantillons, {len(np.unique(y))} classes")
except ValueError as e:
    print(f"Erreur lors de la conversion en NumPy : {e}")
    exit()

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

model = Sequential([
    Bidirectional(LSTM(128, return_sequences=False, dropout=lstm_dropout), 
                  input_shape=(X.shape[1], X.shape[2])),
    Dense(64, activation='relu'),
    Dropout(dropout_rate),
    Dense(len(le.classes_), activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.0005), 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=patience, min_delta=min_delta, 
                               restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=5, min_lr=1e-6)

model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), 
          callbacks=[early_stopping, lr_scheduler], verbose=1)

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Validation Loss: {loss:.4f}, Validation Accuracy: {accuracy:.4f}")

model.save('micro_gestures_model.keras')
print("Modèle entraîné et sauvegardé dans 'micro_gestures_model.keras'")

np.save('label_encoder_classes.npy', le.classes_)
print("Classes de l'encodeur sauvegardées dans 'label_encoder_classes.npy'")