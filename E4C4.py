from __future__ import absolute_import, division, print_function

import collections
import numpy as np
from six.moves import range
import tensorflow as tf
import datetime

from tensorflow_federated import python as tff
from tensorflow.python.keras.optimizer_v2 import gradient_descent
from tensorflow.keras import layers


tf.compat.v1.enable_v2_behavior()

NUM_EXAMPLES_PER_USER = 2000
BATCH_SIZE = 32
USERS = 5
NUM_EPOCHS = 4
CLASSES = 10
WIDTH = 32
HEIGHT = 32
CHANNELS = 3

def mane():
    """ Run program """
    cifar_train, cifar_test = tf.keras.datasets.cifar10.load_data()
    federated_train_data = [get_distributed(cifar_train, u) for u in range(USERS)]
    federated_test_data = [get_distributed(cifar_test, u) for u in range(USERS)]
    sample_batch = federated_train_data[1][-2]
    
    def model_fn():
    	keras_model = create_compiled_keras_model()
    	return tff.learning.from_compiled_keras_model(keras_model, sample_batch)

    iterative_process = tff.learning.build_federated_averaging_process(model_fn)
    evaluation = tff.learning.build_federated_evaluation(model_fn)
    state = iterative_process.initialize()
    fd_test_accuracy = []
    fd_train_loss = []
    for round_num in range(6):
        selected = np.random.choice(5, 4, replace=False)
    	state, metrics = iterative_process.next(state, list(np.array(federated_train_data)[selected]))
    	test_metrics = evaluation(state.model, federated_test_data)
    	fd_train_loss.append(metrics[1])
    	fd_test_accuracy.append(test_metrics.sparse_categorical_accuracy)

    try:
    	with open('Log/Exp3/R24E4C4.txt', 'w') as log:
    		print("Cifar10, Federated E4C4, IDD, minibatch_size: 32", file=log)
    		print("Train Loss: {}".format(fd_train_loss), file=log)
    		print("Test Accuracy: {}".format(fd_test_accuracy), file=log)
    except IOError:
    	print('File Error')

def get_indices_even(y):
    # split dataset into arrays of each class label
    indices_array = []
    for c in range(CLASSES):
        indices_array.append([i for i, d in enumerate(y) if d == c])
    
    user_indices = []
    class_shares = NUM_EXAMPLES_PER_USER // CLASSES
    
    # take even shares of each class for every user
    for u in range(USERS):
        starting_index = u*class_shares
        user_indices.append(np.array(indices_array).T[starting_index:starting_index + class_shares].flatten())   
    return user_indices
    
def get_distributed(source, u):
    indices = get_indices_even(source[1])[u]
    output_sequence = []
    for repeat in range(NUM_EPOCHS):
        for i in range(0, len(indices), BATCH_SIZE):
            batch_samples = indices[i:i + BATCH_SIZE]
            output_sequence.append({
                'x': np.array([source[0][b] / 255.0 for b in batch_samples], dtype=np.float32),
                'y': np.array([source[1][b] for b in batch_samples], dtype=np.int32)})
    return output_sequence


def create_compiled_keras_model():
	model = tf.keras.models.Sequential([
		tf.keras.layers.Conv2D(32,(3, 3),
            activation="tanh",
            padding="same",
            input_shape=(WIDTH, HEIGHT, CHANNELS)),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="tanh", padding="same"),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
        tf.keras.layers.Flatten(), 
        tf.keras.layers.Dense(128, activation="tanh"),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

	def loss_fn(y_true, y_pred):
		return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred))

	model.compile(loss=loss_fn, optimizer="adam", metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
	return model


if __name__ == "__main__":
    
    mane()





