# data utils for DSL 2016 in keras

import numpy as np
from operator import itemgetter
import codecs

train_file = '../data/task2-train.nodup.csv.train'
test_file = '../data/task2-train.nodup.csv.dev'
# use for cross-validation experiments
full_train_file = '../data/task2-train.nodup.csv'
labels_file = '../data/task2-train.labels.txt'
buckwalter = "'|>&<}AbptvjHxd*rzs$SDTZEg_fqklmnhwyYFNKaui~o`{PJVGIOW"
other_symbols = "0123456789-,;.!:/\\@#%^*+-=()[] "
alphabet = buckwalter + other_symbols
task1_train_file = '../data/task1-train.csv'
task1_test_file = '../data/task1-dev.csv'
task1_labels_file = '../data/task1-train.labels.txt'
task1_alphabet_file = '../data/task1-train.alphabet.txt'
task1_B_labels_file = '../data/task1-test-B.labels.txt'

# linguistic features
# features from: http://link.springer.com/chapter/10.1007/978-981-10-0515-2_3
EGYPTIAN_MPCA = {'dh', 'dy', 'dY', 'kd', 'kdA', 'kdh', 'Ayh', '<yh', 'bjd', 'Awy', '>wy', 'AwY', '>wY', 'HSl', 'bqY', 'bq', 'wAd', 'rys', 'fy$', 'd$', 'Hd$', 'm$', 'btAE'}
# union of Jordanian, Palestinian, Syrian
# omitted some prepositions
LEVANTINE_MPCA = {'EAl', 'A$y', '<$y', 'HdA', 'zlm', 'HAy', 'lsA', 'm$', 'rH', 'rAH', 'ly$', '$q', 'hsh', 'hsp', 'mrh', 'm$', '$w', 'hdA', 'hd', 'Em', 'mtl', 'hl}', 'tEy', 'mw', 'ly$', 'hAd', 'hnn', 'HnA', 'EnA'}
# omitted some prefixes
TUNISIAN_MPCA = {'ny$', 'fmA', 'mw$', 'blA', 'bA$', '$n' ,'$nw', '$kw', 'nHb', 'br$p', 'br$h', 'lHq', 'hkA'}
MSA_MPCA = {'swf', 'qd', 'lys', 'lm', 'mA*A', 'lh', 'h*h', '<n', 'qwm', 'ndm', 'wjd', 'tjd'}
# features from Encyclopedia of Arabic Language and Linguistics
# Entries: Gulf States, Kuwaiti, Bahraini, Omani, Moroccan, Algiers, Tunis, Tripoli, Cairo, Jordanian (Amman), Palestinian, Damascus
GULF_STATES_EALL = {'hst', 'm$', 'Akw', '>kw', 'mAkw'}
KUWAITI_EALL = {'AhnA', '<HnA', 'HnA', '$nw', '$nhw', 'mnw', 'mnhw', 'EAd', 'AlHyn', 'lbArHp', 'lbArHh', 'dwm', 'AZHp', 'AZHh', '<ZHp', '<ZHh', 'AljAblh', 'AljAblp', '<ljAblh', '<ljAblp', 'AllAblp', 'AllAblh', '<llAblp', '<llAblh', 'mSyf', '<hnAk', 'AhnAk', 'hSwb', 'wAyd', 'wAjd', 'zyn', 'kl$', 'ztAt', 'wkAd', 'mAl', 'Hq', 'mwb', 'mw', 'mhwb', 'hwb', 'Akw', '<kw', 'mAkw', 'qdAm', 'jdAm', 'mjAbl', 'mqAbl', 'wyA', 'ym', 'dAyr', 'mdAr', 'wyn'}
BAHRAINI_EALL = {'$lwn', 'wyn', '$hst', '$nw', '$nhw', 'wy$', 'wy$w', 'EAd', 'qd', 'kd', 'AlHyn', '<lHyn', '>lHyn', 'AZHp', 'AZHh', '<ZHp', '<ZHh', 'msyAn', 'AljAblh', 'AljAblp', '<ljAblh', '<ljAblp', 'AllAblp', 'AllAblh', '<llAblp', '<llAblh', 'amsyp', 'amsyh', '<hnAk', 'AhnAk', 'hSwb', 'wAjd', 'wAyd', 'hnmwnh', 'hnmwnp', 'kl$', 'zyn', 'ztAt', 'wkAd', 'EsA', 'hrwA', 'Hrwh', 'Hrwp', 'mHAry', 'dydyh', 'mAl', 'Hq', 'mAkw', 'mwb', 'mb', 'hyb', 'mhyb', 'hst', 'mqAbl', 'mjAbl', 'jdAm', 'qdAm', 'Eqb', 'ym', '$klAt', 'lyn'}
OMANI_EALL = {'nHnA', 'AHnA', '<HnA', 'HnA', 'bw', 'Ally', 'hy$', 'wy$', 'A$', '<$', '<y$', '$qd', 'EAd', 'tw', 'AlHyn', '>lHyn', 'dwm', 'nwb', 'nwbp', 'nwbh', 'vnA', 'vnh', 'vnp', 'Eqb', 'xlAf', 'AlbArHh', 'AlbArHp', 'AZHp', 'AZHh', '<ZHp', '<ZHh', 'AlqAylh', 'AlqAylp', '<lqAylh', '<lqAylp', 'nhAryp', 'nhAryh', 'hnAha', 'AhnAk', '<hnAk', 'hst', 'wAjd', 'wAyd', 'zyn', 'hwdAr', 'brAbr', 'kl$', 'qwb', 'ywb', 'dydyh', 'ztAt', 'wAHy', 'Hd', 'mAl', 'Hq', 'HAl', 'mAb', 'wyA', 'qdAm', 'Swb',  'ym'}
GULF_EALL = GULF_STATES_EALL.union(KUWAITI_EALL).union(BAHRAINI_EALL).union(OMANI_EALL)
MOROCCAN_EALL = {'AntAyA', 'AntyyA', 'lly', 'dAbA', 'dAbh', 'dAbp', 'dymA', 'dymh', 'dymp', 'hnAyA', 'tmA', 'tmp', 'tmh', 'tmAk', 'tmAyA', 'tmAyp', 'tmAyh', 'AlqdAm', 'wAlw', 'bzAf', 'qAE', '$HAl', 'dgyp', 'dgyh', 'wAqylA', 'wAqylp', 'wAqylh', 'wA$', '$kwn', '$nw', '$HAl', 'ElA$', 'kyfA$', 'mnAyn', 'fwqA$', 'dyAl', 'd', 'kAyn', 'jwj', 'zwz', 'tsEwd', 'lwkAn', 'kwkAn', 'briz', 'fryn', '$Akmh', '$Akmp', 'kwdlArwT'}
ALGIERS_EALL= {'rA', 'mtE', 'mtAE', 'dyAl', 'rAH', 'Ed'}
TUNIS_EALL = {'db$', 'AHnA', '>HnA', '<ntwmA', 'AntwmA', '>$', '$nwwp', '$nwh', '$nyh', '$nyh', '$nwmh', '$nwmp', '$skwn', 'wyn', 'fyn', 'lwyn', 'mnyn', 'ElA$', 'lwA$', 'kyfA$', 'wqtA$', 'qdA$', 'fmh', 'fmp', 'vmh', 'vmp', 'twp' ,'twh', 'wqthA', 'mnqbl', 'mbEd', 'lbArH', 'gdwh', 'gdwp', 'gAdy', 'gAdykA', 'gAdykp', 'gAdykh', 'hkh', 'hkp', 'qAEd', 'br$p', 'br$h'}
GULF_EALL = GULF_STATES_EALL.union(KUWAITI_EALL).union(BAHRAINI_EALL).union(OMANI_EALL)
MOROCCAN_EALL = {'AntAyA', 'AntyyA', 'lly', 'dAbA', 'dAbh', 'dAbp', 'dymA', 'dymh', 'dymp', 'hnAyA', 'tmA', 'tmp', 'tmh', 'tmAk', 'tmAyA', 'tmAyp', 'tmAyh', 'AlqdAm', 'wAlw', 'bzAf', 'qAE', '$HAl', 'dgyp', 'dgyh', 'wAqylA', 'wAqylp', 'wAqylh', 'wA$', '$kwn', '$nw', '$HAl', 'ElA$', 'kyfA$', 'mnAyn', 'fwqA$', 'dyAl', 'd', 'kAyn', 'jwj', 'zwz', 'tsEwd', 'lwkAn', 'kwkAn', 'briz', 'fryn', '$Akmh', '$Akmp', 'kwdlArwT'}
ALGIERS_EALL= {'rA', 'mtE', 'mtAE', 'dyAl', 'rAH', 'Ed'}
TUNIS_EALL = {'db$', 'AHnA', '>HnA', '<ntwmA', 'AntwmA', '>$', '$nwwp', '$nwh', '$nyh', '$nyh', '$nwmh', '$nwmp', '$skwn', 'wyn', 'fyn', 'lwyn', 'mnyn', 'ElA$', 'lwA$', 'kyfA$', 'wqtA$', 'qdA$', 'fmh', 'fmp', 'vmh', 'vmp', 'twp' ,'twh', 'wqthA', 'mnqbl', 'mbEd', 'lbArH', 'gdwh', 'gdwp', 'gAdy', 'gAdykA', 'gAdykp', 'gAdykh', 'hkh', 'hkp', 'qAEd', 'br$p', 'br$h'}
TRIPOLI_EALL = {'$n', '$nw', '$ynw', '$yny', '$ny', '$kwn', 'f$yn', 'bA$', 'fA$', 'ElA$', 'lA$', 'qdA$', 'wyn', 'mnyn', 'mnwyn', 'hnAyA', 'gAdy', 'gAdykA', 'gAdykAy', 'gAdykAyA', 'lwTA', 'gdwp', 'gdwh', 'dymA', 'dymh', 'dymp', 'bkry', 'fysAE', 'bs', 'HqA', 'dHyp', 'dHyh', 'Hw$'}
NORTH_AFRICAN_EALL = MOROCCAN_EALL.union(ALGIERS_EALL).union(TUNIS_EALL).union(TRIPOLI_EALL).union({'Ally'})
# only Cairo, as other Egyptian dialects are probably not represented in this dataset and might confuse (some are closer to north African)
EGYPTIAN_EALL = {'dh', 'dy', 'dwl', 'dwkhA', 'dykhA', 'dwkhAm', 'Ahw', '>hw', 'Ahy', '>hy', 'Ahwm', '<hwm', '<h', '<mtA', 'AlnhArdp', 'AlnhArdh', 'AnhArdp', 'AnhArdh', 'bkrp', 'brkh', 'dlwqt', 'dlwqty', 'fyn', 'mnyn', 'AzAy', '<zAy', 'Awy', '>wy', 'xAlS', 'kdA', 'kdh', 'kdp', 'btAE', 'btAEp', 'btwE', 'm$', 'kwys', '<zA', 'EAwz', 'HAjp', 'AlHajp', 'bqY', 'bs', 'Ally', 'lmA'}
JORDANIAN_EALL = {'Anw', 'Any', '>nw', '>ny', '$w', 'A$', '<y$', '<ymtA', 'AymtA', 'AmtA', '<mtA', 'bkrh', 'bkrp' ,'mbArH', 'hlA', 'wyn', 'mnwyn', 'hwn', 'hyk', 'ktyr', 'ly$', 'qdy$', '>dy$', 'tbE', 'm$', 'lmA', 'E$An', 'bs', '<zA', 'qAEd', 'Em', 'EmAl'}
PALESTINIAN_EALL = {'AHnA', '<HnA', 'hAy', 'Ally', 'A$', '<$', '$w', 'wkty$', 'wqty$', '<ymtA', 'AymtA', 'AmtA', '<mtA', 'wyn', 'ly$', 'A$y', '<$y', 'hwn', 'hlqyt', 'hyk', 'hy*', 'bkrp', 'bkrh', 'qdAm', 'jmb', 'jnb', 'bs', 'E$An', 'El$An', 'm$', 'rAH', 'Zl', '<zA'}
# only Damascus
SYRIAN_EALL = {'nHn', 'Ally', 'ylly', 'yAlly', '$w', '<$', '<y$', 'hlA', 'mbArH', '<ymtA', 'AymtA', 'AmtA', '<mtA', 'hwn', 'hnyk', 'wyn', 'mnyn', '$lwn', 'ktyr', 'dgry', 'ly$', 'qdy$', 'tbE', 'lmA', 'Em', 'EmAl', 'rAH', '<zA'}
# only Beirut
LEBANON_EALL = {'ylly', '$w', 'hwn', 'hnyk', 'wyn', 'mnyn', 'lwyn', '<ymtA', 'AymtA', 'AmtA', '<mtA', 'ly$', 'tbE', 'm$', 'bs', 'lmA', '$y', 'rH', 'rAH', 'Dl'}
LEVANTINE_EALL = JORDANIAN_EALL.union(PALESTINIAN_EALL).union(SYRIAN_EALL).union(LEBANON_EALL)
# all
EGYPTIAN = sorted(EGYPTIAN_MPCA.union(EGYPTIAN_EALL))
LEVANTINE = sorted(LEVANTINE_MPCA.union(LEVANTINE_EALL))
GULF = sorted(GULF_EALL)
NORTH_AFRICAN = sorted(TUNISIAN_MPCA.union(NORTH_AFRICAN_EALL))
MSA = sorted(MSA_MPCA)

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
    with codecs.open(filename, encoding='utf-8') as f:
        for line in f:
            text = line.strip()
            if text.startswith('"') and text.endswith('"'):
                text = text[1:-1]            
            indices = [char2idx.get(c, 1) for c in text]
            X.append(indices)
    print 'found', len(X), 'examples'
    return np.array(X)  


def load_file(filename, alphabet, shift_idx_to_zero=SHIFT_IDX_TO_ZERO, task=1):
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
    with codecs.open(filename, encoding='utf-8') as f:
        for line in f:
            cls, text = line.strip().split(',', 1)
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
    with codecs.open(filename, encoding='utf-8') as f:
        for line in f:
            cls, text = line.strip().split(',', 1)
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
    with codecs.open(filename, encoding='utf-8') as f:
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

    in_file = codecs.open(filename, encoding='utf-8')
    # first pass, count
    word2count = dict()
    for line in in_file:
        cls, text = line.strip().split(',', 1)
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
        for word in text.split():
            word2count[word] = word2count.get(word, 0)
    in_file.close()
    # second pass, make word2idx
    sorted_word_counts = sorted(word2count.iteritems(), key=itemgetter(1), reverse=True)
    print 'made word2idx with len:', len(word2idx)
    return word2idx


def load_data_ling_feats(train_file, test_file):

    X_train, y_train, num_classes_train = load_file_ling_feats(train_file, alphabet)
    X_test, y_test, _ = load_file_ling_feats(test_file, alphabet)
    return (X_train, y_train), (X_test, y_test), num_classes_train

    

def load_file_ling_feats(filename, shift_idx_to_zero=SHIFT_IDX_TO_ZERO):
    """
    shift_idx_to_zero: if True, assume classes start at 1 and shift to 0
    filename: file in csv format, containing class index and text, 
              where text is surrounded with double quotes
    This version will have a binary vector representing existense of words in lists of indicative words 
    
    returns: X, y, num_classes
    """

    print 'loading data from:', filename
    X, y = [], []
    with codecs.open(filename, encoding='utf-8') as f:
        for line in f:
            cls, text = line.strip().split(',', 1)
            cls = int(cls)
            if shift_idx_to_zero:
                cls -= 1
            if text.startswith('"') and text.endswith('"'):
                text = text[1:-1]
            vec = make_ling_feats_vec(text)
            X.append(vec)
            y.append(cls)
    num_classes = len(np.unique(y))
    print 'found', len(X), 'examples'
    return np.array(X), np.array(y), num_classes    


def load_test_file_ling_feats(filename):
    """
    filename: file containing test examples, one per line, with no labels
    returns: X
    """

    print 'loading data from:', filename
    X = []
    with codecs.open(filename, encoding='utf-8') as f:
        for line in f:
            text = line.strip()
            if text.startswith('"') and text.endswith('"'):
                text = text[1:-1]
            vec = make_ling_feats_vec(text)
            X.append(vec)
    print 'found', len(X), 'examples'
    return np.array(X)


def make_ling_feats_vec(text):
    """ Make a binary vector of linguistic features """

    vec = []
    for dialect in [EGYPTIAN, LEVANTINE, GULF, NORTH_AFRICAN, MSA]:
        for feat in dialect:
            if text.find(' ' + feat + ' ') != -1:
                vec.append(1.0)
            else:
                vec.append(0.0)
    return vec    


def write_predictions_to_file(input_filename, output_filename, predictions, idx2label):
    
    with codecs.open(input_filename, encoding='utf-8') as f_in:
        with codecs.open(output_filename, 'w', encoding='utf-8') as f_out:
            lines = f_in.readlines()
            assert len(lines) == len(predictions), 'incompatible lengths of input file lines and predictions'
            for line, pred in zip(lines, predictions):
                f_out.write(line.strip() + '\t' + idx2label.get(pred, 'ERROR') + '\n')
    print 'written predictions to:', output_filename
    

def write_probabilities_to_file(input_filename, output_filename, probabilities):
    
    with codecs.open(input_filename, encoding='utf-8') as f_in:
        with codecs.open(output_filename, 'w', encoding='utf-8') as f_out:
            lines = f_in.readlines()
            assert len(lines) == len(probabilities), 'incompatible lengths of input file lines and predictions'
            for line, probs in zip(lines, probabilities):
                f_out.write(line.strip() + '\t' + ' '.join([str(prob) for prob in probs]) + '\n')
    print 'written probabilities to:', output_filename


def get_task1_alphabet():

    with codecs.open(task1_alphabet_file, encoding='utf-8') as f:
        alphabet = f.read().strip()
    return alphabet


def predict_with_allowed_labels(probabilities, idx2label, allowed_labels):
    """
    Predict from a set of allowed labels
    
    probabilities: 2d array of examples by label probabilities
    returns: predictions: list of indexes of best allowed labels

    Use in test sets B1/2, where we have a limited set of labels.
    """
    predictions = []
    for probs in probabilities:
        for idx, prob in sorted(enumerate(probs), key=itemgetter(1), reverse=True):
            if idx2label[idx] in allowed_labels:
                predictions.append(idx)
                break
        else:
            sys.stderr.write('Warning: could not find allowed label in predict_with_allowed_labels\n')
    assert len(predictions) == len(probabilities), 'bad predictions in predict_with_allowed_labels\n'
    return predictions

        

