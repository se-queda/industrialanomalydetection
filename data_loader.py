import os
import tensorflow as tf


IMG_SIZE = 256
channel = 3
AUTOTUNE= tf.data.AUTOTUNE


def collect_image_paths(root_dir, mode ="train", categories  = None):
    data = []
    if categories is None:
        categories = os.listdir(root_dir)
    for category in categories:
        folder = os.path.join(root_dir, category)

        if mode == "train":
            img_dir = os.path.join(folder, 'train', 'good')
            for fname in os.listdir(img_dir):
                img_path = os.path.join(img_dir, fname)
                data.append((img_path,0))
        elif mode == "test":
            test_dir = os.path.join(folder, 'test')
            for subfolder in os.listdir(test_dir):
                 defect_dir = os.path.join(test_dir, subfolder)
                 if not os.path.isdir(defect_dir):
                     continue
                 label = 0 if subfolder == "good" else 1
                 for fname in os.listdir(defect_dir):
                    img_path = os.path.join(defect_dir, fname)
                    data.append((img_path, label))
        print(f"[DEBUG] Collected {len(data)} samples for category '{category}' in mode '{mode}'")
    return data


def image_parser(image_path, label):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels = channel, expand_animations=False)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32)/     127.5 - 1.0
    return img, label

def augmentor(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    image = tf.image.adjust_jpeg_quality(image, jpeg_quality=tf.random.uniform([], 90, 100, dtype=tf.int32))
    image = tf.clip_by_value(image, -1.0, 1.0)
    return image, label


def get_dataset(root_dir, categories=None, mode="train", augment=False, batch_size=32):
    samples = collect_image_paths(root_dir, mode, categories)
    path, labels = zip(*samples)

    path = tf.constant(list(path), dtype=tf.string)
    labels = tf.constant(list(labels), dtype=tf.int64)

    ds = tf.data.Dataset.from_tensor_slices((path, labels))
    ds = ds.map(image_parser, num_parallel_calls=AUTOTUNE)

    if augment and mode == "train":
        ds = ds.map(augmentor, num_parallel_calls=AUTOTUNE)

    ds = ds.shuffle(buffer_size=1024)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTOTUNE)

    return ds
