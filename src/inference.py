import os
import argparse
import json
import time
import numpy as np
import tensorflow as tf
import faiss
from scipy.ndimage import gaussian_filter
from PIL import Image
import matplotlib.pyplot as plt

# Import the necessary model components from your src directory
from src.backbone import backbone_model
from src.projector import Projector
from src.patch_aggregator import aggregate
from src.EMAnetwork import EMANetwork

# --- CONFIGURATION ---
IMG_SIZE = 256
SAVE_DIR = "production_assets"
STUDENT_MODEL_WEIGHTS = os.path.join(SAVE_DIR, 'student_model.h5')
MEMORY_BANK_INDEX = os.path.join(SAVE_DIR, 'memory_bank.index')

# --- 1. SET POLICY AND LOAD ASSETS ---
tf.keras.mixed_precision.set_global_policy('mixed_float16')

print("[INFO] Loading production assets for inference...")
# Build the model architecture
backbone = backbone_model()
f_proj = Projector(input_dim=2304, output_dim=128)
g_proj = Projector(input_dim=4608, output_dim=128)
student_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(None, 128)),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(128))
])
# Load the saved weights
student_model.load_weights(STUDENT_MODEL_WEIGHTS)
print("  -> Student model weights loaded.")

# Load the FAISS memory bank
memory_bank_index = faiss.read_index(MEMORY_BANK_INDEX)
# If you trained on a GPU, we need to move the index to the GPU for inference
# For CPU-only inference, you can comment out the next two lines.
res = faiss.StandardGpuResources()
memory_bank_index = faiss.index_cpu_to_gpu(res, 0, memory_bank_index)
print("  -> Memory bank index loaded and moved to GPU.")


# --- 2. DEFINE THE INFERENCE FUNCTION ---
@tf.function
def run_inference_step(image_tensor):
    """A compiled function to run the model forward pass."""
    feat_block2, feat_block3 = backbone(image_tensor, training=False)
    patches_b2, patches_b3 = aggregate(feat_block2, feat_block3)
    z_f = f_proj(patches_b2)
    z_g = g_proj(patches_b3)
    embeddings = tf.concat([z_f, z_g], axis=1)
    student_embeddings = student_model(embeddings, training=False)
    return student_embeddings, (tf.shape(feat_block2)[1], tf.shape(feat_block2)[2])


def inspect_image(image_path):
    """
    Runs the full inspection pipeline on a single image.
    """
    # Pre-process the image
    img = Image.open(image_path).convert('RGB')
    img_resized = img.resize((IMG_SIZE, IMG_SIZE))
    img_tensor = tf.convert_to_tensor(np.array(img_resized), dtype=tf.float32)
    img_tensor = tf.expand_dims(img_tensor, 0)
    img_tensor = img_tensor / 127.5 - 1.0

    # Run the model
    start_time = time.time()
    student_embeddings, patch_shapes = run_inference_step(img_tensor)
    inference_time = time.time() - start_time

    # Calculate Anomaly Score
    student_embeddings_np = student_embeddings.numpy().reshape(-1, 128)
    dists, _ = memory_bank_index.search(student_embeddings_np.astype('float32'), k=1)

    # Generate Anomaly Map
    patch_scores = tf.reshape(dists, (1, patch_shapes[0], patch_shapes[1]))
    anomaly_map = tf.image.resize(patch_scores[:, :, :, tf.newaxis], [IMG_SIZE, IMG_SIZE]).numpy()
    anomaly_map_smoothed = gaussian_filter(anomaly_map[0, :, :, 0], sigma=4)

    image_level_score = np.max(anomaly_map_smoothed)

    # Save heatmap overlay
    heatmap_path = f"heatmap_{os.path.basename(image_path)}"
    plt.imshow(img_resized)
    plt.imshow(anomaly_map_smoothed, cmap='jet', alpha=0.5)
    plt.axis('off')
    plt.savefig(heatmap_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    return {
        "image_path": image_path,
        "anomaly_score": float(image_level_score),
        "is_anomaly": bool(image_level_score > 0.5),  # Example threshold
        "heatmap_path": heatmap_path,
        "inference_time_ms": round(inference_time * 1000, 2)
    }


# --- 3. SCRIPT ENTRY POINT ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run industrial anomaly detection inference.")
    parser.add_argument("image_path", type=str, help="Path to the image to inspect.")
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"[ERROR] Image not found at: {args.image_path}")
    else:
        results = inspect_image(args.image_path)
        print("\n--- INSPECTION RESULTS ---")
        # Print results as a single line of JSON for the Node.js backend
        print(json.dumps(results))