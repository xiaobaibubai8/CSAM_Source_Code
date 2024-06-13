import os
import re


def camel_and_snake_to_word_list(token):
    token = re.sub(r"(?P<key>[A-Z])", r"_\g<key>", token)
    tokens = token.split('_')
    final_token = ''
    first_in = True
    for t in tokens:
        if len(t) > 1:
            final_token += ' ' + t
            first_in = True
        else:
            if first_in:
                final_token += ' ' + t
                first_in = False
            else:
                final_token += t
    return final_token.strip()


def load_stopwords():
    base_dir = os.path.dirname(__file__)
    stopwords_path = os.path.join(base_dir, 'code_stopwords.txt')
    stopwords = set()
    with open(stopwords_path, 'r', encoding='UTF-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.replace('\r', '').replace('\n', '').replace('\t', '').strip()
            if line != '':
                stopwords.add(line)
    return stopwords


if __name__ == '__main__':
    print(camel_and_snake_to_word_list('getResourceCompliantRRAllocation'))
    print(load_stopwords())
    print(len(load_stopwords()))
    print('\'')
