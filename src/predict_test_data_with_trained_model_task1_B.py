""" Code to predict labels on test data with trained model """

from keras.preprocessing import sequence
from keras.models import load_model
from data import load_test_file, write_predictions_to_file, write_probabilities_to_file, load_labels, task1_labels_file, get_task1_alphabet, task1_B_labels_file, predict_with_allowed_labels
alphabet = get_task1_alphabet()

# limit tensorflow memory usage
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
set_session(tf.Session(config=config))


# some hyperparameters (must conform to those in cnn.py)
batch_size = 64
maxlen = 400


# file with only texts, no labels
evaluation_test_file = "../data/B2.txt"
predictions_file = evaluation_test_file + ".large.morehidden.moredrop.pred"
probabilities_file = evaluation_test_file+ ".large.morehidden.moredrop.prob"
model_file = "cnn_model_gpu_multifilter_large_morehidden_moredrop_task1.hdf5"
idx2label = load_labels(task1_labels_file)
allowed_labels = set([l.strip() for l in open(task1_B_labels_file).readlines()])

print 'loading model:', model_file
print('loading model:', model_file)
model = load_model(model_file)

X_test = load_test_file(evaluation_test_file, alphabet)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)

probabilities = model.predict(X_test, batch_size=batch_size)
predictions = predict_with_allowed_labels(probabilities, idx2label, allowed_labels)
write_predictions_to_file(evaluation_test_file, predictions_file, predictions, idx2label)
write_probabilities_to_file(evaluation_test_file, probabilities_file, probabilities)

