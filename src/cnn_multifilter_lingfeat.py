'''Character CNN code for DSL 2016 task 2 

Partly based on: 
https://github.com/fchollet/keras/blob/master/examples/imdb_cnn.py
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
import tensorflow as tf
tf.set_random_seed(1337) # probably not needed

from keras.preprocessing import sequence
from keras.models import Model, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Embedding, merge
from keras.layers import Convolution1D, MaxPooling1D
#from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.utils import np_utils
#from keras.regularizers import l1, l2, l1l2, activity_l1, activity_l2, activity_l1l2
#from keras.layers.normalization import BatchNormalization
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from data import load_data, load_labels, alphabet, train_file, test_file, labels_file, load_data_ling_feats

# limit tensorflow memory usage
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
set_session(tf.Session(config=config))
# set tensorflow random seed for reproducibility

# model file
model_file = "cnn_model_gpu_multifilter_lingfeat.hdf5"

# set parameters:
print('Hyperparameters:')
alphabet_size = len(alphabet) + 2 # add 2, one padding and unknown chars
print('Alphabet size:', alphabet_size)
maxlen = 400
print('Max text len:', maxlen)
batch_size = 16
print('Batch size:', batch_size)
embedding_dims = 50
print('Embedding dim:', embedding_dims)
nb_filters = [50,50,100,100,100,100,100]
print('Number of filters:', nb_filters)
filter_lengths = [1,2,3,4,5,6,7]
print('Filter lengths:', filter_lengths)
hidden_dims = 250
print('Hidden dems:', hidden_dims)
nb_epoch = 30
embedding_droupout = 0.2
print('Embedding dropout:', embedding_droupout)
fc_dropout = 0.5
print('Fully-connected dropout:', fc_dropout)

print('Loading data...')
(X_train, y_train), (X_test, y_test), num_classes = load_data(train_file, test_file, alphabet)
(X_train_lingfeats, _), (X_test_lingfeats, _), _ = load_data_ling_feats(train_file, test_file)
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
X_train_lingfeats = np.array(X_train_lingfeats)
X_test_lingfeats = np.array(X_test_lingfeats)
print('X_train_lingfeats shape:', X_train_lingfeats.shape)
print('X_test_lingfeats shape:', X_test_lingfeats.shape)
num_lingfeats = X_train_lingfeats.shape[1]
print('Number of linguistic features:', num_lingfeats)
y_train = np.array(y_train)
y_test = np.array(y_test)

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, num_classes)
Y_test = np_utils.to_categorical(y_test, num_classes)


print('Build model...')
main_input = Input(shape=(maxlen,))

# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
embedding_layer = Embedding(alphabet_size,
                    embedding_dims,
                    input_length=maxlen,
                    dropout=embedding_droupout)
embedded = embedding_layer(main_input)

# we add a Convolution1D for each filter length, which will learn nb_filters[i]
# word group filters of size filter_lengths[i]:
convs = []
for i in xrange(len(nb_filters)):
    conv_layer = Convolution1D(nb_filter=nb_filters[i],
                        filter_length=filter_lengths[i],
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1)
    conv_out = conv_layer(embedded)
    # we use max pooling:
    conv_out = MaxPooling1D(pool_length=conv_layer.output_shape[1])(conv_out)
    # We flatten the output of the conv layer,
    # so that we can concat all conv outpus and add a vanilla dense layer:
    conv_out = Flatten()(conv_out)
    convs.append(conv_out)

# concat all conv outputs
x = merge(convs, mode='concat') if len(convs) > 1 else convs[0]
#concat = BatchNormalization()(concat)

# We add a vanilla hidden layer:
x = Dense(hidden_dims)(x)
x = Dropout(fc_dropout)(x)
x = Activation('relu')(x)

# now add linguistic features
ling_input = Input(shape=(num_lingfeats,))
x_ling = ling_input
#x_ling = Dense(num_lingfeats)(ling_input)
#x_ling = Dropout(fc_dropout)(x_ling)
#x_ling = Activation('relu')(x_ling)

merged = merge([x, x_ling], mode='concat')
#merged = x_ling

# We project onto number of classes output layer, and squash it with a softmax:
main_output = Dense(num_classes, activation='softmax')(merged)

# finally, define the model 
model = Model(input=[main_input, ling_input], output=main_output)
#model = Model(input=ling_input, output=main_output)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
print('Train...')
# define callbacks
stopping = EarlyStopping(monitor='val_loss', patience='10')
checkpointer = ModelCheckpoint(filepath=model_file, verbose=1, save_best_only=True)
tensorboard = TensorBoard(log_dir="./logs-multifilter-lingfeat", write_graph=False)
model.fit([X_train, X_train_lingfeats], Y_train,
#model.fit(X_train_lingfeats, Y_train,
          batch_size=batch_size,
          nb_epoch=nb_epoch,
          validation_data=([X_test, X_test_lingfeats], Y_test), 
#          validation_data=(X_test_lingfeats, Y_test), 
          callbacks=[stopping, checkpointer, tensorboard])
          
probabilities = model.predict([X_test, X_test_lingfeats], batch_size=batch_size)
#probabilities = model.predict(X_test_lingfeats, batch_size=batch_size)
predictions = probabilities.argmax(axis=-1)
idx2label = load_labels(labels_file)
#with open('cnn_predictions.txt', 'w') as g:
#    for i in xrange(len(y_test)):
#        g.write(' '.join([str(v) for v in X_test[i]]) + '\t' + idx2label.get(y_test[i], 'ERROR') + '\t' + idx2label.get(predictions[i], 'ERROR') + '\n')
print('Performance of final model (not necessarily best model):')
print('========================================================')
cm = confusion_matrix(y_test, predictions)
print('Confusion matrix:')
print(cm)
acc = accuracy_score(y_test, predictions)
print('Accuracy score:')
print(acc)
labels = [label for (idx, label) in sorted(idx2label.items())]
score_report = classification_report(y_test, predictions, target_names=labels)
print('Score report:')
print(score_report)
best_model = load_model(model_file)
probabilities = best_model.predict([X_test, X_test_lingfeats], batch_size=batch_size)
#probabilities = best_model.predict(X_test_lingfeats, batch_size=batch_size)
predictions = probabilities.argmax(axis=-1)
print('Performance of best model:')
print('==========================')
cm = confusion_matrix(y_test, predictions)
print('Confusion matrix:')
print(cm)
acc = accuracy_score(y_test, predictions)
print('Accuracy score:')
print(acc)
labels = [label for (idx, label) in sorted(idx2label.items())]
score_report = classification_report(y_test, predictions, target_names=labels)
print('Score report:')
print(score_report)
