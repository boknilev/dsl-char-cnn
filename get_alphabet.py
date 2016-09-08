import codecs, sys


def get_alphabet(csv_filename):

    alphabet = set()
    with codecs.open(csv_filename, encoding='utf-8') as f_in:
        for line in f_in:
            label, text = line.strip().split(',', 1)
            if text.startswith('"') and text.endswith('"'):
                text = text[1:-1]
            for c in text:
                alphabet.add(c)
    return alphabet


def run(csv_filename, alphabet_output_filename):

    alphabet = get_alphabet(csv_filename)
    for c in alphabet:
        print c, '|', 
    print
    with codecs.open(alphabet_output_filename, 'w', encoding='utf8') as f_out:
        for c in sorted(alphabet):
            print c
            try:
                f_out.write(c)
            except UnicodeDecodeError:
                sys.stderr.write('Warning: caught UnicodeDecodeError for charactter:\n')
                sys.stderr.write(c)
        f_out.write('\n')


if __name__ == '__main__':
    if len(sys.argv) == 3:
        run(sys.argv[1], sys.argv[2])
    else:
        print 'USAGE: python ' + sys.argv[0] + ' <csv file> <alphabet output file>'



