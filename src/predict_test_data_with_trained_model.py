""" Code to predict labels on test data with trained model """

from keras.preprocessing import sequence
from keras.models import load_model
from data import load_test_file, write_predictions_to_file, write_probabilities_to_file, load_labels, labels_file, alphabet

# limit tensorflow memory usage
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
set_session(tf.Session(config=config))


# some hyperparameters (must conform to those in cnn.py)
batch_size = 16
maxlen = 400


# file with only texts, no labels
evaluation_test_file = "../data/C.txt"
predictions_file = evaluation_test_file + ".fulltrain.pred"
probabilities_file = evaluation_test_file+ ".fulltrain.prob"
model_file = "cnn_model_gpu_multifilter_fulltrain.hdf5"
idx2label = load_labels(labels_file)

model = load_model(model_file)

X_test = load_test_file(evaluation_test_file, alphabet)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)

probabilities = model.predict(X_test, batch_size=batch_size)
predictions = probabilities.argmax(axis=-1)
write_predictions_to_file(evaluation_test_file, predictions_file, predictions, idx2label)
write_probabilities_to_file(evaluation_test_file, probabilities_file, probabilities)

