# src/extract_areas_10k.py

import pandas as pd
import numpy as np
import h5py
import tensorflow as tf

# ==================================================
# CONFIG
# ==================================================
INPUT_CSV   = "Datasets/train_30k.csv"
MODEL_PATH  = "models/segmentation_model_30k.h5"
OUTPUT_CSV  = "Datasets/areas_10k_tumor_only.csv"

IMG_SIZE = 128
ORIG_SIZE = 240
THRESHOLD = 0.3

TARGET_SAMPLES = 10000
BATCH_SIZE = 16
RANDOM_SEED = 42

# ==================================================
# CUSTOM OBJECTS
# ==================================================
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth
    )

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return bce + dice_loss(y_true, y_pred)

# ==================================================
# LOAD MODEL
# ==================================================
model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "dice_coef": dice_coef,
        "bce_dice_loss": bce_dice_loss
    }
)

print("Segmentation model loaded.")

# ==================================================
# LOAD FILES
# ==================================================
df = pd.read_csv(INPUT_CSV)
df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

file_list = df["filepath"].tolist()

print("Available pool:", len(file_list))

# ==================================================
# PREPROCESS
# ==================================================
def load_image(file_path):
    with h5py.File(file_path, "r") as f:
        img = f["image"][:]

    img = img[:, :, :4]

    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE)).numpy()
    img = img.astype(np.float32)

    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    return img

# ==================================================
# MAIN LOOP
# ==================================================
results = []
checked = 0

for start in range(0, len(file_list), BATCH_SIZE):
    if len(results) >= TARGET_SAMPLES:
        break

    batch_files = file_list[start:start+BATCH_SIZE]
    X_batch = []

    for file_path in batch_files:
        X_batch.append(load_image(file_path))

    X_batch = np.array(X_batch)
    preds = model.predict(X_batch, verbose=0)

    for i, pred in enumerate(preds):
        if len(results) >= TARGET_SAMPLES:
            break

        pred_mask = (pred > THRESHOLD).astype(np.uint8)

        pred_mask_240 = tf.image.resize(
            pred_mask,
            (ORIG_SIZE, ORIG_SIZE),
            method="nearest"
        ).numpy()

        pred_mask_240 = pred_mask_240[:, :, 0].astype(np.uint8)

        area_pixels = int(np.sum(pred_mask_240))

        checked += 1

        # Keep only tumor samples
        if area_pixels > 0:
            area_mm2 = area_pixels
            area_cm2 = area_mm2 / 100

            results.append([
                batch_files[i],
                area_pixels,
                round(area_mm2, 2),
                round(area_cm2, 2)
            ])

    if checked % 500 == 0 or len(results) >= TARGET_SAMPLES:
        print(
            f"Checked {checked} | "
            f"Tumor samples collected: {len(results)}"
        )

# ==================================================
# SAVE CSV
# ==================================================
out_df = pd.DataFrame(
    results,
    columns=[
        "filepath",
        "area_pixels",
        "area_mm2",
        "area_cm2"
    ]
)

out_df.to_csv(OUTPUT_CSV, index=False)

print("\nSaved:", OUTPUT_CSV)
print("Final tumor-only samples:", len(out_df))
print(out_df.head())