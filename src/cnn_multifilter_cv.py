'''CNN code for DSL 2016 task 2, with cross validation
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Embedding, merge
from keras.layers import Convolution1D, MaxPooling1D
#from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import np_utils
#from keras.regularizers import l1, l2, l1l2, activity_l1, activity_l2, activity_l1l2
#from keras.layers.normalization import BatchNormalization
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.cross_validation import StratifiedKFold
from data import load_file, load_labels, alphabet, full_train_file, labels_file

# limit tensorflow memory usage
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
set_session(tf.Session(config=config))


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
#nb_filters = [50,50,100,100,100,100,100]
nb_filters = [80,80,80]
print('Number of filters:', nb_filters)
#filter_lengths = [1,2,3,4,5,6,7]
filter_lengths = [4,5,6]
print('Filter lengths:', filter_lengths)
hidden_dims = 250
print('Hidden dems:', hidden_dims)
nb_epoch = 20
embedding_droupout = 0.2
print('Embedding dropout:', embedding_droupout)
fc_dropout = 0.5
print('Fully-connected dropout:', fc_dropout)

# cross validation
n_folds = 10

print('Loading data...')
X_train, y_train, num_classes = load_file(full_train_file, alphabet)
print(len(X_train), 'train sequences')

print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
print('X_train shape:', X_train.shape)
y_train = np.array(y_train)

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, num_classes)


def make_model(maxlen, alphabet_size, embedding_dims, embedding_droupout,
               nb_filters, filter_lengths, hidden_dims, fc_dropout, 
               num_classes):
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
    
    # We project onto number of classes output layer, and squash it with a softmax:
    main_output = Dense(num_classes, activation='softmax')(x)
    
    # finally, define the model 
    model = Model(input=main_input, output=main_output)
    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    return model
    
def train_and_evaluate_model(model, X_train, Y_train, X_test, Y_test, y_test, fold):
    # y_test is labels, Y_test is categorical labels
    
    print('Train...')
    stopping = EarlyStopping(monitor='val_loss', patience='10')
    model_filename = "cnn_model_gpu_multifilter_fold{}.hdf5".format(fold)
    checkpointer = ModelCheckpoint(filepath=model_filename, verbose=1, save_best_only=True)
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              validation_data=(X_test, Y_test), 
              callbacks=[stopping,checkpointer],
              verbose=2)
    probabilities = model.predict(X_test, batch_size=batch_size)
    predictions = probabilities.argmax(axis=-1)
    acc = accuracy_score(y_test, predictions)
    print('Accuracy score (final model): {}'.format(acc))
    best_model = load_model(model_filename)
    probabilities = best_model.predict(X_test, batch_size=batch_size)
    predictions = probabilities.argmax(axis=-1)
    best_acc = accuracy_score(y_test, predictions)
    print('Accuracy score (best model): {}'.format(best_acc))
    return best_acc
    

# run cross validation (based on: https://github.com/fchollet/keras/issues/1711)
skf = StratifiedKFold(y_train, n_folds=n_folds, shuffle=True)
accuracies = []
for k, (train, test) in enumerate(skf):
    print("Running fold {}/{}".format(k+1, n_folds))
    model = None # clearing the NN
    model = make_model(maxlen, alphabet_size, embedding_dims, embedding_droupout,
               nb_filters, filter_lengths, hidden_dims, fc_dropout, 
               num_classes)
    acc = train_and_evaluate_model(model, X_train[train], Y_train[train], X_train[test], Y_train[test], y_train[test], k+1)
    accuracies.append(acc)
print('Accuracies of all folds:')
print(accuracies)

    
    


