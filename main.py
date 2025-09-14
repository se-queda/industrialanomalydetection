import os
import tensorflow as tf
import time
from src.data_loader import get_dataset
from src.backbone import backbone_model
from src.projector import Projector
from src.EMAnetwork import EMANetwork
from src.trainer import train

ROOT_DIR = "/home/utsab/Downloads/mvtec_anomaly_detection"
categories = os.listdir(ROOT_DIR)

start_time = time.time()

train_ds = get_dataset(ROOT_DIR, categories=categories, mode="train", augment=False, batch_size=1)
test_ds = get_dataset(ROOT_DIR, categories=categories, mode="test", augment=False, batch_size=1)


backbone = backbone_model()
f_proj = Projector(input_dim=2304, output_dim=128)
g_proj = Projector(input_dim=4608, output_dim=128)

student_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(None, 128)),     # (N,128) per sample
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(128))
])
ema_model = EMANetwork(student_model, momentum=0.99)

print("[INFO] Starting training...")
memory_bank = train(
    train_ds,
    backbone,
    f_proj,
    g_proj,
    student_model,
    ema_model,
    epochs=1,
    lr=1e-3,
    coreset_size=5000
)

end_time = time.time()
print(f"[RESULT] Data loading + preprocessing time + feature extraction + patch extration + 1d projection + EMAnetwork+ contrastive loss +memory bank + corsetsampling: {end_time - start_time:.2f} seconds")
