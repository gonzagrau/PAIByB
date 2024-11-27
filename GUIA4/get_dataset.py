import os
import gdown
import tensorflow as tf
from matplotlib import pyplot as plt
from typing import Tuple, List, Any, Dict

URL_TEMPLATE = r'https://drive.google.com/uc?id={}'
DATASET_URL = URL_TEMPLATE.format('1FzsBjfNG6Njt7YnXLc_bgSEb11AEYOQD')
TFRECORD_FILE = 'train_img_mask_label_id.tfrecord'


def parse_example(example_proto: Any) -> Dict[str, tf.Tensor | str] :
    """
    Parse a single example from a TFRecord file.

    Args:
        example_proto (record): A single record from a TFRecord file.

    Returns:
        image (tf.Tensor): The image data.
        mask (tf.Tensor): The mask data.
        label (tf.Tensor): The label data.
        image_id (tf.Tensor): The image ID.
    """
    features = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "mask": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.string),
        "image_id": tf.io.FixedLenFeature([], tf.string),
    }
    parsed_features = tf.io.parse_single_example(example_proto, features)
    parsed_features["image"] = tf.io.decode_jpeg(parsed_features["image"])
    parsed_features["mask"] = tf.io.decode_png(parsed_features["mask"], channels=1)  # Decode PNG to retain single-channel mask
    return parsed_features


def load_dataset(tfrecord_file: str = TFRECORD_FILE) -> tf.data.TFRecordDataset:
    """
    Load a dataset from a TFRecord file.
    If the file does not exist, it will be downloaded.

    Args:
        tfrecord_file (str): The path to the TFRecord file.

    Returns:
        dataset (tf.data.TFRecordDataset): The dataset.
    """
    if not os.path.exists(tfrecord_file):
        gdown.download(DATASET_URL, tfrecord_file, quiet=False)

    dataset = tf.data.TFRecordDataset(tfrecord_file, compression_type="GZIP")
    dataset = dataset.map(parse_example)
    return dataset


def main():
    dataset = load_dataset()
    dataset = dataset.shuffle(buffer_size=1000, reshuffle_each_iteration=True)

    # Show an example
    example_labels = []
    example_ids = []

    for record in dataset.take(5):
        image, mask, label, image_id = record["image"], record["mask"], record["label"], record["image_id"]
        image = image.numpy()
        mask = mask.numpy()
        label = label.numpy().decode('utf-8')
        example_labels.append(label)
        image_id = image_id.numpy().decode('utf-8')
        example_ids.append(image_id)
        
        plt.imshow(image)
        plt.imshow(mask, alpha=0.2, cmap='Blues')
        plt.title(f"Image {image_id}, {label=}")
        plt.axis('off')
        plt.show()

if __name__ == '__main__':
    main()
 