import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import sys

# Load the trained model
model = tf.keras.models.load_model('models/cnn_model.h5')

# Define CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Function to load and preprocess a single image
def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(32, 32))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize
    return img_array

# Main prediction logic
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py <path_to_image>")
        sys.exit(1)

    img_path = sys.argv[1]
    img_data = prepare_image(img_path)

    predictions = model.predict(img_data)
    predicted_class = class_names[np.argmax(predictions)]

    print(f"Predicted Class: {predicted_class}")
