from data_loader import get_dataset
from backbone import backbone_model
import tensorflow as tf
from patch_aggregator import aggregate
import time
import os

ROOT_DIR = "/home/utsab/Downloads/mvtec_anomaly_detection"
categories = os.listdir(ROOT_DIR)

start_time = time.time()

train_ds = get_dataset(ROOT_DIR, categories=categories, mode="train", augment=False, batch_size=8)  # Use smaller batch for test
test_ds = get_dataset(ROOT_DIR , categories = categories, mode = "test", augment = False, batch_size = 8)


# 🧠 Get one batch from train_ds
for batch_images, batch_labels in train_ds.take(1):
    print(f"[INFO] Input batch shape: {batch_images.shape}")  # Should be (8, 256, 256, 3)

    # 🛠️ Load backbone model
    backbone = backbone_model()

    # 🔍 Run inference
    feat_block2, feat_block3 = backbone(batch_images)

    print(f"[INFO] Block2 output shape: {feat_block2.shape}")   # Expected: (8, 64, 64, 256)
    print(f"[INFO] Block3 output shape: {feat_block3.shape}")   # Expected: (8, 32, 32, 512)

    break
patches_b2, patches_b3 = aggregate(feat_block2, feat_block3)

print(f"[INFO] Patches Block2: {patches_b2.shape}")  # (B, num_patches, patch_dim)
print(f"[INFO] Patches Block3: {patches_b3.shape}")
end_time = time.time()
print(f"[RESULT] Data loading + preprocessing time + feature extraction + patch extration: {end_time - start_time:.2f} seconds")
