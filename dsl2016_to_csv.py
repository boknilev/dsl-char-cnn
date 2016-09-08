# Convert a DSL 2016 file (task 2) to a csv file


import sys, codecs


def get_labels(dsl_filename, task=2):

    assert task == 2 or task == 1, 'unknown task: ' + str(task) + '\n'
    labels = set()
    with open(dsl_filename) as f:
        for line in f:
            if line.strip() == '':
                continue
            if task == 2:
                _, _, label = line.strip().split('\t')
            else:
                _, label = line.strip().split('\t')
            labels.add(label)
    print 'labels:', labels
    return list(labels)


def convert_file(dsl_filename, csv_filename, label2id, task=2):

    assert task == 2 or task == 1, 'unknown task: ' + str(task) + '\n'
    encoding = 'utf-8' if task == 1 else None
    with codecs.open(dsl_filename, encoding=encoding) as f:
        with codecs.open(csv_filename, 'w', encoding=encoding) as g:
            for line in f:
                if task == 2:
                    text, _, label = line.strip().split('\t')
                else:
                    text, label = line.strip().split('\t')
                assert label in label2id, 'unknown label: ' + label + '\n'
                g.write(str(label2id[label]) + ',' + '"' + text + '"' + '\n')


def run(dsl_filename, csv_filename, labels_filename, task=2):

    assert task == 2 or task == 1, 'unknown task: ' + str(task) + '\n'
    labels = get_labels(dsl_filename, task)
    with open(labels_filename, 'w') as g:
        g.write('\n'.join(labels) + '\n')
    label2id = dict(zip(labels, range(1, len(labels)+1)))
    convert_file(dsl_filename, csv_filename, label2id, task)


if __name__ == '__main__':
    if len(sys.argv) == 4:
        run(sys.argv[1], sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 5:
        run(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]))
    else:
        print 'USAGE: python ' + sys.argv[0] + ' <dsl file> <csv file> <labels file> [<task>]'


