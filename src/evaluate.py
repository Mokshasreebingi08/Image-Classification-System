import tensorflow as tf
from load_data import load_data

# Load the model
model = tf.keras.models.load_model('models/cnn_model.h5')

# Load Data
(_, _), (test_images, test_labels) = load_data()

# Normalize test images
test_images = test_images / 255.0

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'Test accuracy: {test_acc}')
