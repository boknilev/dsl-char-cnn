""" Code to predict labels on test data with trained model """

import os
import numpy as np
from keras.preprocessing import sequence
from keras.models import load_model
from data import load_test_file, write_predictions_to_file, write_probabilities_to_file, load_labels, labels_file, alphabet
from collections import Counter

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
predictions_file = evaluation_test_file + ".cvtrain.pred"
probabilities_file = evaluation_test_file+ ".cvtrain.prob"
model_files_pref = "cnn_model_gpu_multifilter_fold"
idx2label = load_labels(labels_file)

# load test data
print('Loading data')
X_test = load_test_file(evaluation_test_file, alphabet)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('Found', len(X_test), 'examples')

# predict with all models
all_probs = []
for model_file in os.listdir('.'):
    if model_file.startswith(model_files_pref) and model_file.endswith('hdf5'):
        print('Predicting with model:', model_file)
        model = load_model(model_file)
        probabilities = model.predict(X_test, batch_size=batch_size)
        prob_file= model_file + '.' + os.path.basename(evaluation_test_file) + '.prob'
        print('Writing probabilities to file:', prob_file)
        write_probabilities_to_file(evaluation_test_file, prob_file, probabilities)
        all_probs.append(probabilities)

# majority vote
print('Getting majority vote')
all_preds = np.argmax(all_probs, axis=2)
# for every example, find the most common prediction
majority_preds = [Counter(all_preds[:,j]).most_common(1)[0][0] for j in xrange(len(X_test))]
write_predictions_to_file(evaluation_test_file, predictions_file, majority_preds, idx2label)


