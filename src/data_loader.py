import os
import tensorflow as tf

IMG_SIZE = 256
CHANNELS = 3
AUTOTUNE = tf.data.AUTOTUNE


def collect_image_paths(root_dir, mode="train", categories=None):
    data = []
    if categories is None:
        categories = os.listdir(root_dir)

    for category in categories:
        folder = os.path.join(root_dir, category)

        if mode == "train":
            img_dir = os.path.join(folder, 'train', 'good')
            for fname in os.listdir(img_dir):
                img_path = os.path.join(img_dir, fname)
                # For training, use an empty string as a placeholder for the mask path
                data.append((img_path, 0, ""))

        elif mode == "test":
            test_dir = os.path.join(folder, 'test')
            mask_dir = os.path.join(folder, 'ground_truth')
            for subfolder in os.listdir(test_dir):
                defect_dir = os.path.join(test_dir, subfolder)
                if not os.path.isdir(defect_dir):
                    continue
                label = 0 if subfolder == "good" else 1
                for fname in os.listdir(defect_dir):
                    img_path = os.path.join(defect_dir, fname)
                    # Use an empty string for good samples with no mask
                    if subfolder == "good":
                        mask_path = ""
                    else:
                        mask_fname = fname.replace('.png', '_mask.png')
                        mask_path = os.path.join(mask_dir, subfolder, mask_fname)

                    data.append((img_path, label, mask_path))

        print(f"[DEBUG] Collected {len(data)} samples for category '{category}' in mode '{mode}'")
    return data


def image_parser(image_path, label, mask_path):
    # Load and process the image
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=CHANNELS)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32) / 127.5 - 1.0  # Normalize to [-1, 1]

    # Conditionally load the mask based on the path
    def read_mask():
        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_png(mask, channels=1)
        mask = tf.image.resize(mask, [IMG_SIZE, IMG_SIZE], method='nearest')
        return tf.cast(mask > 0, dtype=tf.uint8)

    def empty_mask():
        return tf.zeros([IMG_SIZE, IMG_SIZE, 1], dtype=tf.uint8)

    # Use tf.cond for conditional logic inside a tf.function
    mask = tf.cond(tf.equal(mask_path, ""), empty_mask, read_mask)

    return img, label, mask


def augmentor(image, label, mask):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    image = tf.clip_by_value(image, -1.0, 1.0)
    return image, label, mask


def get_dataset(root_dir, categories=None, mode="train", augment=False, batch_size=32, cache=True):
    paths, labels, mask_paths = zip(*collect_image_paths(root_dir, mode, categories))

    ds = tf.data.Dataset.from_tensor_slices((list(paths), list(labels), list(mask_paths)))

    if mode == "train":
        ds = ds.shuffle(buffer_size=len(paths))

    ds = ds.map(image_parser, num_parallel_calls=AUTOTUNE)

    if cache:
        ds = ds.cache()

    if augment and mode == "train":
        ds = ds.map(augmentor, num_parallel_calls=AUTOTUNE)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTOTUNE)

    return ds