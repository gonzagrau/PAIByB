import os
import gdown
import tensorflow as tf
from typing import Tuple, List, Any


URL_TEMPLATE = r'https://drive.google.com/uc?id={}'
DATASET_URL = URL_TEMPLATE.format('1qakYj0HsbH-823ooJpxJ5SHldLumBdK4')


def parse_example(example_proto: Any) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
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
    image = tf.io.decode_jpeg(parsed_features["image"])
    mask = tf.io.decode_png(parsed_features["mask"], channels=1)  # Decode PNG to retain single-channel mask
    label = parsed_features["label"]
    image_id = parsed_features["image_id"]
    return image, mask, label, image_id


def load_dataset(tfrecord_file):
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
    dataset = tf.data.TFRecordDataset(tfrecord_file)
    dataset = dataset.map(parse_example)
    return dataset


def main():
    tfrecord_file = 'dataset.tfrecord'
    dataset = load_dataset(tfrecord_file)
    dataset = dataset.shuffle(buffer_size=1000, reshuffle_each_iteration=True)

    # Show an example
    example_labels = []
    example_ids = []

    for image, mask, label, image_id in dataset.take(5):
        image = image.numpy()
        mask = mask.numpy()
        label = label.numpy().decode('utf-8')
        example_labels.append(label)
        image_id = image_id.numpy().decode('utf-8')
        example_ids.append(image_id)
        
        plt.imshow(image)
        plt.imshow(mask, alpha=0.2, cmap='Reds')
        plt.title(f"Image {image_id}, {label=}")
        plt.axis('off')
        plt.show()

if __name__ == '__main__':
    #main()
 