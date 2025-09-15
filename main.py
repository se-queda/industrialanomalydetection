import os
import tensorflow as tf
import time
import faiss  # Make sure to import faiss
from src.data_loader import get_dataset
from src.backbone import backbone_model
from src.projector import Projector
from src.EMAnetwork import EMANetwork
from src.trainer import train
from src.evaluate import evaluate

# --- 1. SET THE MIXED PRECISION POLICY ---
tf.keras.mixed_precision.set_global_policy('mixed_float16')

ROOT_DIR = "/home/utsab/Downloads/mvtec_anomaly_detection"
SAVE_DIR = "production_assets"  # A directory to store your final assets
os.makedirs(SAVE_DIR, exist_ok=True)

# Using a subset for faster training
categories = ['toothbrush']

start_time = time.time()

train_ds = get_dataset(ROOT_DIR, categories=categories, mode="train", augment=False, batch_size=16)
test_ds = get_dataset(ROOT_DIR, categories=categories, mode="test", augment=False, batch_size=16)


backbone = backbone_model()
f_proj = Projector(input_dim=2304, output_dim=128)
g_proj = Projector(input_dim=4608, output_dim=128)

student_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(None, 128)),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(128))
])
ema_model = EMANetwork(student_model, momentum=0.99)

print("[INFO] Starting training...")

# --- 2. CREATE THE OPTIMIZER AND WRAP IT ---
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)


# --- 3. RUN TRAINING ---
memory_bank = train(
    train_ds,
    backbone,
    f_proj,
    g_proj,
    student_model,
    ema_model,
    optimizer,
    epochs=10, # Using 10 epochs as a practical number
    coreset_size=5000
)

print("\n[INFO] Training complete. Starting test validation...")

# --- 4. RUN TEST VALIDATION ---
image_auroc, pixel_auroc = evaluate(
    test_ds,
    backbone,
    f_proj,
    g_proj,
    student_model,
    memory_bank
)

print(f"\n--- FINAL METRICS ---")
print(f"Image-level AUROC: {image_auroc:.4f}")
print(f"Pixel-level AUROC: {pixel_auroc:.4f}")

# --- 5. SAVE PRODUCTION ASSETS ---
print("\n[INFO] Saving production assets...")
student_model_weights_path = os.path.join(SAVE_DIR, 'student_model.h5')
memory_bank_index_path = os.path.join(SAVE_DIR, 'memory_bank.index')

student_model.save_weights(student_model_weights_path)
faiss.write_index(memory_bank.index, memory_bank_index_path)

print(f"  -> Student model weights saved to: {student_model_weights_path}")
print(f"  -> Memory bank index saved to: {memory_bank_index_path}")


end_time = time.time()
print(f"\n[RESULT] Total script time: {end_time - start_time:.2f} seconds")