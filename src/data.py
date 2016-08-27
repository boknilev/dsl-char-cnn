# data utils for DSL 2016 in keras

import numpy as np
from operator import itemgetter


train_file = '../data/task2-train.nodup.csv.train'
test_file = '../data/task2-train.nodup.csv.dev'
# use for cross-validation experiments
full_train_file = '../data/task2-train.nodup.csv'
labels_file = '../data/task2-train.labels.txt'
buckwalter = "'|>&<}AbptvjHxd*rzs$SDTZEg_fqklmnhwyYFNKaui~o`{PJVGIOW"
other_symbols = "0123456789-,;.!:/\\@#%^*+-=()[] "
alphabet = buckwalter + other_symbols

SHIFT_IDX_TO_ZERO = True


def load_labels(labels_file, shift_idx_to_zero=SHIFT_IDX_TO_ZERO):
    idx2label = dict()
    with open(labels_file) as f:
        idx = 0 if shift_idx_to_zero else 1
        for line in f:
            idx2label[idx] = line.strip()
            idx += 1
    return idx2label


def load_data(train_file, test_file, alphabet):

    X_train, y_train, num_classes_train = load_file(train_file, alphabet)
    X_test, y_test, _ = load_file(test_file, alphabet)
    return (X_train, y_train), (X_test, y_test), num_classes_train


def load_test_file(filename, alphabet):
    """
    Load test data
    
    filename: file containing test examples, one per line, with no labels
    alphabet: string containing characters to use, all others are mapped to 1
              (0 saved for padding)    
    """
    
    print 'loading test data from:', filename
    char2idx = dict(zip(list(alphabet), range(2, len(alphabet)+2)))
    X = []
    with open(filename) as f:
        for line in f:
            text = line.strip()
            if text.startswith('"') and text.endswith('"'):
                text = text[1:-1]            
            indices = [char2idx.get(c, 1) for c in text]
            X.append(indices)
    print 'found', len(X), 'examples'
    return np.array(X)  


def load_file(filename, alphabet, shift_idx_to_zero=SHIFT_IDX_TO_ZERO):
    """
    shift_idx_to_zero: if True, assume classes start at 1 and shift to 0
    filename: file in csv format, containing class index and text, 
              where text is surrounded with double quotes
    alphabet: string containing characters to use, all others are mapped to 1
              (0 saved for padding)
    """

    print 'loading data from:', filename
    char2idx = dict(zip(list(alphabet), range(2, len(alphabet)+2)))
    X, y = [], []
    with open(filename) as f:
        for line in f:
            cls, text = line.strip().split(',')
            cls = int(cls)
            if shift_idx_to_zero:
                cls -= 1
            if text.startswith('"') and text.endswith('"'):
                text = text[1:-1]
            indices = [char2idx.get(c, 1) for c in text]
            X.append(indices)
            y.append(cls)
    num_classes = len(np.unique(y))
    print 'found', len(X), 'examples'
    return np.array(X), np.array(y), num_classes


def load_data_word_vecs(train_file, test_file, max_words=5000):

    word2idx = make_word2idx(train_file, max_words)
    X_train, y_train, num_classes_train = load_file_word_vecs(train_file, word2idx)
    X_test, y_test, _ = load_file_word_vecs(test_file, word2idx)
    return (X_train, y_train), (X_test, y_test), num_classes_train



def load_file_word_vecs(filename, word2idx, shift_idx_to_zero=SHIFT_IDX_TO_ZERO):
    """
    shift_idx_to_zero: if True, assume classes start at 1 and shift to 0
    filename: file in csv format, containing class index and text, 
              where text is surrounded with double quotes
    word2idx: dictionary from word to index
    returns: X, y, num_classes
    """

    print 'loading data from:', filename
    X, y = [], []
    with open(filename) as f:
        for line in f:
            cls, text = line.strip().split(',')
            cls = int(cls)
            if shift_idx_to_zero:
                cls -= 1
            if text.startswith('"') and text.endswith('"'):
                text = text[1:-1]
            indices = [word2idx.get(word, 1) for word in text.split()]
            X.append(indices)
            y.append(cls)
    num_classes = len(np.unique(y))
    print 'found', len(X), 'examples'
    return np.array(X), np.array(y), num_classes


def load_test_file_word_vecs(filename, word2idx):
    """
    filename: file containing test examples, one per line, with no labels
    word2idx: dictionary from word to index. Must be provided during testing
    returns: X
    """

    print 'loading data from:', filename
    X = []
    with open(filename) as f:
        for line in f:
            text = line.strip()
            if text.startswith('"') and text.endswith('"'):
                text = text[1:-1]
            indices = [word2idx.get(word, 1) for word in text.split()]
            X.append(indices)
    print 'found', len(X), 'examples'
    return np.array(X)


def make_word2idx(filename, max_words=5000):
    """ Make word to idx dictionary for data preparation

    filename: file in csv format, containing class index and text, 
              where text is surrounded with double quotes
    max_words: maximum number of most frequent words to include
    """

    in_file = open(filename)
    # first pass, count
    word2count = dict()
    for line in in_file:
        cls, text = line.strip().split(',')
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
        for word in text.split():
            word2count[word] = word2count.get(word, 0)
    in_file.close()
    # second pass, make word2idx
    sorted_word_counts = sorted(word2count.iteritems(), key=itemgetter(1), reverse=True)
    word2idx = dict()
    for word, count in sorted_word_counts[:max_words]:
        # reserve 0 and 1 for padding and unknown word
        word2idx[word] = len(word2idx) + 2
    print 'made word2idx with len:', len(word2idx)
    return word2idx


def write_predictions_to_file(input_filename, output_filename, predictions, idx2label):
    
    with open(input_filename) as f_in:
        with open(output_filename, 'w') as f_out:
            lines = f_in.readlines()
            assert len(lines) == len(predictions), 'incompatible lengths of input file lines and predictions'
            for line, pred in zip(lines, predictions):
                f_out.write(line.strip() + '\t' + idx2label.get(pred, 'ERROR') + '\n')
    print 'written predictions to:', output_filename
    

def write_probabilities_to_file(input_filename, output_filename, probabilities):
    
    with open(input_filename) as f_in:
        with open(output_filename, 'w') as f_out:
            lines = f_in.readlines()
            assert len(lines) == len(probabilities), 'incompatible lengths of input file lines and predictions'
            for line, probs in zip(lines, probabilities):
                f_out.write(line.strip() + '\t' + ' '.join([str(prob) for prob in probs]) + '\n')
    print 'written probabilities to:', output_filename

