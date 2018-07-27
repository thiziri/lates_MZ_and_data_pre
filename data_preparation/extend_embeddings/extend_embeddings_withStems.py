# coding: utf8
# extending a word embedding with its stemmed vocabulary
from __future__ import print_function

import sys
import six
import io
import array
import numpy as np
from tqdm import tqdm
import pickle
from tools4text import check_word
import mimetypes
from gensim.models import KeyedVectors as Word2Vec


"""
Pros:
Read a vocabulary of a given glove w2v_file
Args:
w2v_file: file, path to file of pre-trained word2vec/glove/fasttext
Returns:
word2embedding: dict
"""
def load_glove_word_embedding(w2v_file):
    print ("Reading as glove model ...")
    pre_trained = {} # word2embedding dict

    # str call is necessary for Python 2/3 compatibility, since
    # argument must be Python 2 str (Python 3 bytes) or
    # Python 3 str (Python 2 unicode)
    vectors, dim = array.array(str('d')), None

    # Try to read the whole file with utf-8 encoding.
    binary_lines = False
    try:
        with io.open(w2v_file, encoding="utf8") as f:
            lines = [line for line in f]
    # If there are malformed lines, read in binary mode
    # and manually decode each word from utf-8
    except:
        print("Could not read {} as UTF8 file, "
              "reading file as bytes and skipping "
              "words with malformed UTF8.".format(w2v_file))
        with open(w2v_file, 'rb') as f:
            lines = [line for line in f]
        binary_lines = True

    print("Loading vectors from {}".format(w2v_file))

    for line in tqdm(lines):
        # Explicitly splitting on " " is important, so we don't
        # get rid of Unicode non-breaking spaces in the vectors.
        entries = line.rstrip().split(b" " if binary_lines else " ")

        word, entries = entries[0], entries[1:]
        if dim is None and len(entries) > 1:
            dim = len(entries)

        elif len(entries) == 1:
            print("Skipping token {} with 1-dimensional "
                  "vector {}; likely a header".format(word, entries))
            continue
        elif dim != len(entries):
            raise RuntimeError(
                "Vector for token {} has {} dimensions, but previously "
                "read vectors have {} dimensions. All vectors must have "
                "the same number of dimensions.".format(word, len(entries), dim))

        if binary_lines:
            try:
                if isinstance(word, six.binary_type):
                    word = word.decode('utf-8')

            except:
                print("Skipping non-UTF8 token {}".format(repr(word)))
                continue

        if word not in pre_trained:
            pre_trained[word] = [float(x) for x in entries]

    return pre_trained

"""
Pros:
Read a vocabulary of a given w2v_file
Args:
w2v_file: file, path to file of pre-trained word2vec/glove/fasttext
Returns:
word2embedding: dict
"""
def load_word2vec_word_embedding(w2v_file):
    print("Reading as word2vec model ...")
    mime = mimetypes.guess_type(w2v_file)
    model = Word2Vec.load_word2vec_format(w2v_file, binary=mime)
    pre_trained = {} # word2embedding dict
    for word in tqdm(model.vocab):
        if word not in pre_trained:
            pre_trained[word] = model[word].tolist()
    del model
    return pre_trained

if __name__ == '__main__':

    w2v_file = sys.argv[1]  # w2v_file
    w2v_withStems_file = sys.argv[2]  # new embedding

    word2stem = {}
    as_glove = False

    print('load word vectors ...')
    try:
        embeddings = load_glove_word_embedding(w2v_file)
        as_glove = True
    except:
        embeddings = load_word2vec_word_embedding(w2v_file)

    print('stemming ...')
    for w in tqdm(embeddings):
        for wc in check_word(w, "krovetz"):
            if wc in word2stem:
                continue
            else:
                try:
                    word2stem [wc] = embeddings [wc]
                except:
                    word2stem [wc] = embeddings [w]

    print('extending ...')
    for w in tqdm(word2stem):
        if w in embeddings:
            continue
        else:
            embeddings[w] = word2stem[w]

    print('save word vectors ...')
    with open(w2v_withStems_file, 'w') as fw:
        # write word_embedings
        if not as_glove:
            fw.write("{vocab_size} {embed_size}\n".format(vocab_size=str(len(embeddings)), embed_size=str(len(list(embeddings.values())[0]))))
        for w in tqdm(embeddings):
            print(w, ' '.join(map(str, embeddings[w])), file=fw)

    print('Stemming word vectors finished.')

