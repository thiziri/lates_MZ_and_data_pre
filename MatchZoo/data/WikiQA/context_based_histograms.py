#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function
import os
import sys
sys.path.append('../../matchzoo/utils/')
sys.path.append('../../matchzoo/inputs/')
from preprocess import cal_hist_context
from rank_io import *

if __name__ == '__main__':
    hist_size = int(sys.argv[1])
    win_size = int(sys.argv[2])
    srcdir = sys.argv[3]
    embedfile = srcdir + 'embed_glove_d300_norm'
    corpusfile = srcdir + 'corpus_preprocessed.txt'

    relfiles = [srcdir + 'relation_train.txt',
                srcdir + 'relation_valid.txt',
                srcdir + 'relation_test.txt'
                ]
    histfiles = [
            srcdir + 'context_relation_train.hist-%d.txt' % hist_size,
            srcdir + 'context_relation_valid.hist-%d.txt' % hist_size,
            srcdir + 'context_relation_test.hist-%d.txt' % hist_size
            ]
    embed_dict = read_embedding(filename=embedfile)
    print('read embedding finished ...')
    _PAD_ = len(embed_dict)
    embed_size = len(list(embed_dict.values())[0])
    embed_dict[_PAD_] = np.zeros((embed_size, ), dtype=np.float32)
    embed = np.float32(np.random.uniform(-0.2, 0.2, [_PAD_+1, embed_size]))
    embed = convert_embed_2_numpy(embed_dict, embed = embed)

    corpus, _ = read_data(corpusfile)
    print('read corpus finished....')
    for idx, relfile in enumerate(relfiles):
        histfile = histfiles[idx]
        rel = read_relation(relfile)
        fout = open(histfile, 'w')
        for label, d1, d2 in rel:
            assert d1 in corpus
            assert d2 in corpus
            qnum = len(corpus[d1])
            d1_embed = embed[corpus[d1]]  # get all embedded words of d1 and d2: sub-matrices embed[[w1, w2 ...]]
            d2_embed = embed[corpus[d2]]
            print(len(d1_embed), len(d2_embed))
            curr_hist = cal_hist_context(d1_embed, d2_embed, qnum, hist_size, win_size)  # compute the context based histogram
            curr_hist = curr_hist.tolist()
            fout.write(' '.join(map(str, curr_hist)))
            fout.write('\n')
        fout.close()
    print('generate histogram finished ...')
