# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
Tools for text extraction and analysis
"""

import collections
import os
from collections import defaultdict
from os import listdir
from os.path import join
from nltk.stem.porter import PorterStemmer
from krovetzstemmer import Stemmer
import re
import nltk
from tqdm import tqdm
import ntpath
import csv
import gzip


"""
It return the file name extracted from a path
"""
def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

"""
removes the file extension:
example: file.txt becomes file
return: file name without extension
"""
def remove_extension(file):
    return file.split('.')[0]


"""
Cleans the input text of special characters
return cleaned text
"""
def escape(input):
    return input.translate({
        ord('('): None,
        ord(')'): None,
        ord('\''): None,
        ord('\"'): None,
        ord('.'): ' ',
        ord(':'): ' ',
        ord('\t'): ' ',
        ord('/'): ' ',
        ord('&'): ' ',
        ord(','): ' ',
        ord('^'): ' ',
        ord('-'): ' ',
        ord('?'): ' ',
        ord('!'): ' ',
        ord('+'): ' ',
        ord(';'): ' ',
        ord('`'): None,
        ord('$'): None,
        ord('€'): None,
        ord('<'): ' ',
        ord('>'): ' ',
        ord('%'): ' ',
        ord('#'): ' ',
        ord('_'): ' ',
        ord('@'): ' ',
        ord('~'): ' ',
        ord('='): None,
        ord('*'): None,
    })


"""
Performs stemming according to the selected algo
return stemed text
"""
def stem(algo, text):
    if algo == "krovetz":
        stemmer = Stemmer()
        return stemmer.stem(text)
    elif algo == "porter":
        stm = PorterStemmer()
        return stm.stem(text)
    print("ERROR STEMMING: {t} unkown.".format(t=algo))


"""
Performs cleaning and stemming 
return cleaned and stemmed text
"""
def clean(text_to_clean, steming, stoplist):
    prog = re.compile("[_\-\(]*([A-Z]\.)*[_\-\(]*")
    tex = []
    for w in text_to_clean.split():
        if prog.match(w):
            w = w.replace('.', '')
        tex.append(w)
    text = " ".join(tex)
    text = ' '.join(escape(text).split())
    text = " ".join(nltk.word_tokenize(text))
    text = " ".join([stem(steming, w) for w in text.split() if w not in stoplist])
    return text


""" 
Extract TREC million queries on the path_top parameter as dictionnary. 
return: dictionnary of queries.
ex: {0:"this is a text of the query"}
"""
def extract_trec_million_queries(path_top):
    topics = {}

    def extract(f):
        print("Processing file ", f)
        if ".gz" not in f:
            input_ = open(join(path_top, f), 'r')  # Reading file
        else:
            input_ = gzip.open(join(path_top, f))
        for line in tqdm(input_.readlines()):
            l = line.decode("iso-8859-15")
            query = l.strip().split(":")
            q = str(int(query[0]))
            q_text = query[-1]  # last token string
            topics[q] = q_text

        return collections.OrderedDict(sorted(topics.items()))

    if os.path.isfile(path_top):
        return extract(path_top)
    else:
        for fi in listdir(path_top):
            topics.update(extract(fi))


"""
Read the qrels file to a dictionary.
Return dictionary of: {(q_id, d_id):rel} 
"""
def get_qrels_1(qrels_file):
        print("Reading Qrels ... ")
        qdr = {}
        with open(qrels_file, 'r') as qrels:
            for line in tqdm(qrels):
                if line is not None:
                    q = str(int(line.strip().split()[0]))
                    doc = line.strip().split()[2]
                    rel = int(line.strip().split()[3])
                    qdr[(q, doc)] = rel
        print("Qrels ok.")
        return collections.OrderedDict(sorted(qdr.items()))


"""
<<<<<<< HEAD
Read the MQ set file to a dictionary and a set of labels.
Return dictionary of: {q_id: {d_id:rel, ...}, ...} 
=======
Read the qrels file to a dictionary.
Return dictionary of: {(q_id, d_id):rel} 
>>>>>>> 994bd944e350343b5f056039e7c2e20270fd589f
"""
def get_qrels(qrels_file):
        labels = set()
        print("Reading Qrels ... ")
        qdr = defaultdict(dict)
        with open(qrels_file, 'r') as qrels:
            for line in tqdm(qrels):
                if line is not None:
                    q = line.strip().split()[1].split(':')[-1]
                    doc = line.strip().split("#docid")[-1].split()[1]
                    qdr[q][doc] = int(line.strip().split()[0])
                    labels.add(qdr[q][doc])
        print("Qrels ok.")
        return collections.OrderedDict(sorted(qdr.items())), labels


"""
Save a collection to corpus.txt file
Return: number of written lines nl
"""
def save_corpus(queries_text, ranked_documents, index, id2token, externelDocId,out):
    print("Saving text corpus ...")
    nl = 0
    for q in collections.OrderedDict(sorted(queries_text.items())):
        out.write("{d} {d_txt}\n".format(d=q, d_txt=queries_text[q], encoding='utf8'))
        nl += 1
    
    for doc in tqdm(ranked_documents):
        try:
            doc_text = " ".join([id2token[x] for x in index.document(externelDocId[doc])[1] if x!=0])
        except:
            doc_text = ""
        if doc_text != "":
            out.write("{d} {d_txt}\n".format(d=doc, d_txt=doc_text, encoding='utf8'))
            nl += 1
    return nl


"""
Read the word_dict.txt file generated by MatchZoo.
Return: word_dict dictionary.
"""
def read_word_dict(word_dict_file):
    with open(word_dict_file, mode='r') as infile:
        reader = csv.reader(infile, delimiter=' ')
        word_dict = {int(rows[1]): rows[0] for rows in reader}
    return word_dict


"""
read unique values from column n in the file f
"""
def read_values(f, n):
    inf = open(f, "r")
    lines = inf.readlines()
    result = []
    for x in lines:
        result.append(x.split()[n])
    inf.close()
    return set(result)


"""
devide list seq into num different sub-lists
return: list of folds
"""
def chunkIt(seq, num=5):
    avg = len(seq) / float(num)
    out = []
    last = 0.0
    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg
    return out


def get_all_phrases(target_word, tar_passage, left_margin=10, right_margin=10):
    """
        Get all the phrases that contain the target word in a text tar_passage.
        Workaround to save the output given by nltk Concordance function.
        left_margin and right_margin allocate the number of words/pununciation before and after target word.

        :param target_word: str
        :param tar_passage: str
        :param left_margin: int
        :param right_margin: int
        :return: list
        """

    # Create list of tokens using nltk function
    tokens = nltk.word_tokenize(tar_passage)

    # Create the text of tokens
    text = nltk.Text(tokens)

    # Collect all the index or offset position of the target word
    c = nltk.ConcordanceIndex(text.tokens, key=lambda s: s.lower())

    # Collect the range of the words that is within the target word by using text.tokens[start;end].
    # The map function is used so that when the offset position - the target range < 0, it will be default to zero
    concordance_txt = ([text.tokens[list(map(lambda x: x - left_margin if (x - left_margin) > 0 else 0, [offset]))[
                                        0]:offset + right_margin if (
                                                                                offset - left_margin) > 0 else offset + right_margin + abs(
        offset - left_margin)]
                        for offset in c.offsets(target_word)])

    # join the sentences for each of the target phrase and return it
    return [' '.join([x + ' ' for x in con_sub]) for con_sub in concordance_txt]


def get_text_of_a_passage(doc_id, index, id2token, passage):
    """
    Get the text corresponding to a retrieved passage from indri
    :param doc_id: int
    :param id2token: list
    :param passage: list
    :return: string
    """
    doc = [x for x in index.document(doc_id)[1]]
    passage_txt = " ".join([id2token[x] for x in doc[passage[0]:passage[1]] if x != 0])
    return passage_txt
