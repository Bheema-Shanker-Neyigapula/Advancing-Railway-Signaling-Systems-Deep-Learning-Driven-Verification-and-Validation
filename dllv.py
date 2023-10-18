import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('trained_model.h5')

# Function to load and preprocess the validation dataset
def load_validation_data():
	validation_data = np.load('validation_data.npy')
	preprocessed_data = preprocess(validation_data)
	return preprocessed_data

# Load and preprocess the validation dataset
validation_data = load_validation_data()

# Evaluate the model on the validation dataset
validation_loss, validation_accuracy = model.evaluate(validation_data, verbose=2)

# Print the validation metrics
print('Validation Loss:', validation_loss)
print('Validation Accuracy:', validation_accuracy)