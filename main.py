import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.layers import TFSMLayer

# ===== CONFIG =====
MODEL_PATH = r"C:\Users\frank\Native Projects\HAM10K\model_saved"
SAMPLE_IMAGE_PATH = r"C:\Users\frank\Native Projects\HAM10K\OIP.jpg"
IMG_SIZE = 224
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

# ===== LOAD MODEL =====
print(f"Loading model from: {MODEL_PATH}")
model = tf.keras.Sequential([TFSMLayer(MODEL_PATH, call_endpoint="serving_default")])
print("✅ Model loaded\n")
model.summary()

# ===== LOAD & PREPROCESS IMAGE =====
img = cv2.imread(SAMPLE_IMAGE_PATH)
if img is None:
    raise FileNotFoundError(f"Image not found: {SAMPLE_IMAGE_PATH}")
img = cv2.cvtColor(cv2.resize(img, (IMG_SIZE, IMG_SIZE)), cv2.COLOR_BGR2RGB) / 255.0
img_batch = np.expand_dims(img, axis=0)

# ===== PREDICT =====
pred_probs = np.array(list(model.predict(img_batch).values())[0])[0]
pred_index = np.argmax(pred_probs)

print(f"\nPrediction: {CLASSES[pred_index]} ({pred_probs[pred_index]*100:.2f}%)")

# ===== DISPLAY RESULTS =====
plt.figure(figsize=(4, 4))
plt.imshow(img)
plt.title(f"{CLASSES[pred_index]} ({pred_probs[pred_index]*100:.2f}%)")
plt.axis("off")

plt.figure(figsize=(8, 4))
bars = plt.bar(CLASSES, pred_probs, color="skyblue")
bars[pred_index].set_color("orange")
plt.title("Prediction Probabilities")
plt.ylabel("Probability")
plt.ylim([0, 1])
for i, prob in enumerate(pred_probs):
    plt.text(i, prob + 0.01, f"{prob*100:.1f}%", ha="center")
plt.show()
