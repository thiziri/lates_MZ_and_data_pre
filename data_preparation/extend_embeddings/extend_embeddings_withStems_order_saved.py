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
from tools4text import path_leaf, stem, line_prepender
import mimetypes
from os.path import join
import os
import re

def check_word_unique(word, stemming):
    prog1 = re.compile("([A-Z]\.)+")
    if (prog1.match(word)):
        w=word.replace('.','')
        word = w
    return stem(stemming, word)

def do_add(s, x):
    return len(s) != (s.add(x) or len(s))

"""
Pros:
Read a vocabulary of a given embedding w2v_file,then perform extension
Args:
w2v_file: file, path to file of pre-trained word2vec/glove/fasttext
Returns:
resulting_file, resulting_vocab_size
"""
def extend_word_embedding(w2v_file, out_dir):
    print ("Reading embeddings ...")
    out_file = join(out_dir, os.path.splitext(path_leaf(w2v_file))[0]+"_extended_with_stems.txt")
    pre_trained = set() # word2embedding dict

    # str call is necessary for Python 2/3 compatibility, since
    # argument must be Python 2 str (Python 3 bytes) or
    # Python 3 str (Python 2 unicode)
    vectors, dim = array.array(str('d')), None

    # Try to read the whole file with utf-8 encoding.
    binary_lines = False
    try:
        f = io.open(w2v_file, encoding="utf8")
        # If there are malformed lines, read in binary mode
        # and manually decode each word from utf-8
    except:
        print("Could not read {} as UTF8 file, "
              "reading file as bytes and skipping "
              "words with malformed UTF8.".format(w2v_file))
        f = open(w2v_file, 'rb')
        binary_lines = True
    
    # for word2vec
    if mimetypes.guess_type(w2v_file)[0]=='application/octet-stream':
        f = open(w2v_file, 'rb')
        binary_lines = True

    print("Loading vectors from {}".format(w2v_file))

    emb_out = open(out_file, "w")
    vocab_size = 0

    for line in tqdm(f):
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
                    #print (word)

            except:
                print("Skipping non-UTF8 token {}".format(repr(word)))
                continue

        if word not in pre_trained:
            vec = [float(x) for x in entries]
            print(word.lower(), ' '.join([str(i) for i in vec]), file=emb_out)
            vocab_size +=1
            pre_trained.add(word.lower())
            stem_word = check_word_unique(word, "krovetz")
            if do_add(pre_trained, stem_word):
                #pre_trained.add(stem_word)
                print(stem_word, ' '.join([str(i) for i in vec]), file=emb_out)
                vocab_size +=1
            else:
                continue

    #line_prepender(out_file, "{V} {dim}".format(V=vocab_size, dim=dim))
    return out_file, vocab_size

if __name__ == '__main__':

    # python3 extend_embeddings_withStems_order_saved.py '/home/thiziri/Documents/DOCTORAT/osirim_data/projets/iris/PROJETS/WEIR/collections/GoogleNews-vectors-negative300.bin' /home/thiziri/Desktop/

    w2v_file = sys.argv[1]  # w2v_file
    out_dir = sys.argv[2]  # output folder

    print('extending ...')
    out_file, vocab_size = extend_word_embedding(w2v_file, out_dir)

    print('Stemming word vectors finished.')
    print('Result in {f} containning {v} embedded words'.format(f=out_file, v=vocab_size))
