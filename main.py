import os
import tensorflow as tf
import time
from src.data_loader import get_dataset
from src.backbone import backbone_model
from src.projector import Projector
from src.EMAnetwork import EMANetwork
from src.trainer import train

# --- 1. SET THE MIXED PRECISION POLICY ---
# This tells TensorFlow to use float16 for computations and float32 for variables.
tf.keras.mixed_precision.set_global_policy('mixed_float16')

ROOT_DIR = "/home/utsab/Downloads/mvtec_anomaly_detection"
categories = os.listdir(ROOT_DIR)

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
# We create the optimizer first...
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
# ...and then wrap it. This wrapper handles the necessary scaling to prevent
# numerical issues that can happen with float16.
optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)


# --- 3. PASS THE OPTIMIZER TO THE TRAIN FUNCTION ---
# We need to modify the 'train' function to accept the optimizer we just created.
memory_bank = train(
    train_ds,
    test_ds,
    backbone,
    f_proj,
    g_proj,
    student_model,
    ema_model,
    optimizer,  # Pass the new optimizer here
    epochs=1,
    coreset_size=5000
)

end_time = time.time()
print(f"[RESULT] Total script time: {end_time - start_time:.2f} seconds")