# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
Tools for text extraction and analysis
"""

import collections
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
from random import shuffle


"""
Write at the beginning of a file.
"""
class Prepender:

    def __init__(self, fname, mode='w'):
        self.__write_queue = []
        self.__f = open(fname, mode)

    def write(self, s):
        self.__write_queue.insert(0, s)

    def close(self):
        self.__exit__(None, None, None)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        if self.__write_queue: 
            self.__f.writelines(self.__write_queue)
        self.__f.close()


"""
Add a line to the beginning of a file.
"""
def line_prepender(filename, line):
    with open(filename, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(line.rstrip('\r\n') + '\n' + content)


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
Performs stemming and dot "." removal for a word 
return checked and stemmed word
"""
def check_word(word, stemming):
    prog1 = re.compile("([A-Z]\.)+")
    prog2 = re.compile("([A-Z]|[a-z])+(\_|\-)+([A-Z]|[a-z])*")
    if prog1.match(word):
        w=word.replace('.','')
        return [stem(stemming, w)]
    if prog2.match(word):
        words = [stem(stemming, w) for w in word.replace('_', ' ').replace('-', ' ').split()]
        return words
    return [stem(stemming, word)]


""" 
Extract TREC topics on the pathTop parameter as dictionnary. 
return dictionnary of queries.
ex: {0:"this is a text of the topic"}
"""
def extractTopics(path_top):
    print("Extraction de : %s" % path_top)
    nb = 0
    topics = {}
    for f in listdir(path_top):
        f = open(join(path_top,f), 'r')   # Reading file
        l = f.readline().lower()
        # extracting topics
        while l != "":
            if l != "":
                num = 0
                while (l.startswith("<num>") == False) and (l != ""):
                    l = f.readline().lower()
                num = l.replace("<num>", "").replace("number:", "").replace("\n", "").replace(" ", "")
                while (l.startswith("<title>")==False) and (l!=""):
                    l = f.readline().lower()
                titre = ""
                while (not l.startswith("</top>")) and (not l.startswith("<desc>")) and (l!=""):
                    titre = titre + " " + l.replace("<title>", "")
                    l = f.readline().lower()
                if titre != "" and num != 0:
                    topics[str(int(num))] = titre.replace("\n", "").replace("topic:", "").replace("\t", " ")
                    nb += 1
            else: 
                print("Fin.\n ")
        f.close()
    return collections.OrderedDict(sorted(topics.items()))


""" 
Extract TREC million queries on the path_top parameter as dictionnary. 
return: dictionnary of queries.
ex: {0:"this is a text of the query"}
"""
def extract_trec_million_queries(path_top):
    topics = {}
    for f in listdir(path_top):
        print("Processing file ", f)
        if ".gz" not in f:
            input = open(join(path_top, f), 'r')   # Reading file
        else:
            input = gzip.open(join(path_top, f))
        for line in tqdm(input.readlines()):
            l = line.decode("iso-8859-15")
            query = l.strip().split(":")
            q = "mq" + str(int(query[0]))
            q_text = query[-1]  # last token string
            topics[q] = q_text
    return collections.OrderedDict(sorted(topics.items()))


"""
Read the qrels file to a dictionary.
Return dictionary of: {(q_id, d_id):rel} 
"""
def get_qrels(qrels_file):
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
Computes a list of relevant judged documents in the trec-like qrels_file
return lists of relevant documents for each query
"""
def relDocs_perQuery(qrels_file):
    print("Relevant documents per-query ...")
    q_relDoc = {}
    with open(qrels_file) as qrels:
        for l in qrels:
            if l.strip().split()[0] not in q_relDoc:
                q_relDoc[l.strip().split()[0]] = []
            if int(l.strip().split()[3])==1:
                q_relDoc[l.strip().split()[0]].append(l.strip().split()[2])
    return q_relDoc


"""
Computes a list of judged documents per query in the trec-like qrels_file
return lists of documents for each query
"""
def docs_perQuery(qrels_file):
    print("Relevant documents per-query ...")
    q_relDoc = {}
    with open(qrels_file) as qrels:
        for l in qrels:
            if l.strip().split()[0] not in q_relDoc:
                q_relDoc[l.strip().split()[0]] = []
            q_relDoc[l.strip().split()[0]].append(l.strip().split()[2])
    return q_relDoc


"""
Read document list from a trec like run_file.
return: a set of ranked distinct documents
"""
def get_docs_from_run(run_file):
    print("Reading run_file: ", run_file)
    docs = []
    with open (run_file) as rf:
        for l in rf:
            if l is not None :
                docs.append(l.strip().split()[2])
    return set(docs)


"""
Add documents from the ranked ones to the training set qrels_equi that's constructed from the qrels file.
Return: qrels_equi padded to qrels_equi2
"""
def pad(qrels_equi, run_file):
    print("Read results per query in the run_file {} for padding ...", run_file)
    qrels_equi2 = qrels_equi
    docs_perQ = {}
    with open (run_file) as rf:
        for l in rf:
            if l!= None:
                q = int(l.strip().split()[0])
                if q in docs_perQ:
                    docs_perQ[q].append(l.strip().split()[2])
                else:
                    docs_perQ[q] = [l.strip().split()[2]]
    num_res = len(docs_perQ[list(docs_perQ.keys())[0]])
    # qrels_equi will be padded to reach this size of results per query

    print("Padding query results ...")
    for q in tqdm(docs_perQ):
        num_inQrels = len([e[0] for e in qrels_equi2 if e[0] == q])
        to_add = num_res - num_inQrels
        docs_q = docs_perQ[q]
        shuffle(docs_q)
        for d in docs_q: # add documents from retrieved ones per a query
            if (q, d) not in qrels_equi2:
                qrels_equi2[(q, d)] = 0
                to_add -=1
            if to_add == 0:
                break
    return qrels_equi2


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
Create relations from a rank file.
 Return: list of relations
"""
def rank2relations(rank_file, if_bin, out):
    relations = []
    with open(rank_file,"r") as rank:
        i = 0
        j = -1
        queries_rank = []
        for line in tqdm(rank):
            if line != None:
                j+=1
                q = line.strip().split()[0]
                if q in queries_rank:
                    i += 1
                else:
                    queries_rank.append(q)
                    i = 1
                doc = line.strip().split()[2]
                #score = line.strip().split()[4]
                if not if_bin:
                    rel = 3 if i<=10 else 2 if (i in range(11,21)) else 1 if (i in range(21,51)) else 0 # multiscale relevance
                    out.write("{q} 0 {d} {r}\n".format(q=q, d=doc, r=rel))
                    if i in range(1, 51) or i in range(901, 1001): # save tuples
                        relations.insert(j, (rel, q, doc))
                else:
                    rel = 1 if i<=10 else 0 # binary relevance
                    out.write("{q} 0 {d} {r}\n".format(q=q, d=doc, r=rel))
                    if i in range(1, 11) or i in range(981, 1001):
                        relations.insert(j, (rel, q, doc))
    # return enumerate(relations)
    return relations


"""
construct a list of relevance judgements associated to each rank interval, 
then gives the corresponding relevance judgement to the given rank
"""
def rank_to_relevance(rank, scales=3, ranks=[[1, 10], [11, 30], [31, 50]]):
    relevance = {(ranks[i][0],ranks[i][1]):scales-i for i in range(len(ranks))}
    for interval in relevance:
        if rank in range(interval[0], interval[1]+1):
            return relevance[interval]


"""
Create relations from a run file. Same as rank2relations but with another format
Return: list of relations [((q, doc), rel)]
"""
def run2relations(run_file, if_bin, qrels, scales, ranks, k=1000):
    relations = []
    with open(run_file, "r") as rank:
        i = 0
        j = -1
        queries_rank = []
        for line in tqdm(rank):
            if line is not None:
                j += 1
                q = str(int(line.strip().split()[0]))
                if q in queries_rank:
                    i += 1
                else:
                    queries_rank.append(q)
                    i = 1
                doc = line.strip().split()[2]
                if len(qrels) == 0:
                    if not if_bin:
                        x = rank_to_relevance(i, scales, ranks)
                        rel = x if x is not None else 0  # multiscale relevance
                    else:
                        rel = 1 if i <= 10 else 0  # binary relevance
                else:
                    try:
                        rel = qrels[(q, doc)]
                    except:
                        rel = 0
                if i in range(k+1):
                    relations.insert(j, ((q, doc), rel))
                else:
                    continue
    return relations


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
            result.append(x.split(' ')[n])
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
