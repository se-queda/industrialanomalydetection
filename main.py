from data_loader import get_dataset
import matplotlib.pyplot as plt
import os


ROOT_DIR = "/home/utsab/Downloads/mvtec_anomaly_detection"
categories = os.listdir(ROOT_DIR)


train_ds = get_dataset(ROOT_DIR, categories=categories, mode="train", augment=False, batch_size=32)
test_ds = get_dataset(ROOT_DIR , categories = categories, mode = "test", augment = False, batch_size = 32)


for images, labels in train_ds.take(1):
    for i in range(8):
        img = (images[i].numpy() + 1.0) * 127.5  # De-normalize [-1,1] → [0,255]
        img = img.astype("uint8")
        plt.subplot(2, 4, i+1)
        plt.imshow(img)
        plt.title(f"Label: {labels[i].numpy()}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()