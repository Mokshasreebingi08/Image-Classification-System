import tensorflow as tf
from load_data import load_data
from model import create_model

# Load Data
(train_images, train_labels), (test_images, test_labels) = load_data()

# Normalize the images to [0, 1]
train_images, test_images = train_images / 255.0, test_images / 255.0

# Create Model
model = create_model()
    
# Train the Model
model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# Save the model
model.save('models/cnn_model.h5')
