import tensorflow as tf
from tensorflow.keras.datasets import cifar10

def load_data():
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    return (train_images, train_labels), (test_images, test_labels)
