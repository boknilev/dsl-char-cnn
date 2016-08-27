# split data set to train/dev

import sys, random


def split_file(input_filename, train_filename, dev_filename, fraction=0.1):

    with open(input_filename) as f:
        with open(train_filename, 'w') as g_train:
            with open(dev_filename, 'w') as g_dev:
                for line in f:
                    if random.random() > 0.1:
                        g_train.write(line)
                    else:
                        g_dev.write(line)


if __name__ == '__main__':
    if len(sys.argv) == 2:
        split_file(sys.argv[1], sys.argv[1] + '.train', sys.argv[1] + '.dev')
    elif len(sys.argv) == 3:
        split_file(sys.argv[1], sys.argv[1] + '.train', sys.argv[1] + '.dev', float(sys.argv[2]))
    else:
        print 'USAGE: python ' + sys.argv[0] + ' <input file> <output train file> <output dev file> [<fraction>]'


