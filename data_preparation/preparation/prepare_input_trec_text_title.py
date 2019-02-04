import pyndri
from os.path import join
import os
import codecs
import sys
import json
import logging
from tqdm import tqdm
from nltk.corpus import stopwords
from collections import defaultdict

sys.path.append('../utils')

from tools4text import extractTopics, clean, get_qrels


logging.basicConfig(filename='collect2MZinpuText.log', level=logging.DEBUG)

if __name__ == '__main__':
    config_file = sys.argv[1]
    config = json.load(open(config_file))
    logging.info('Config: '+json.dumps(config, indent=2))

    print("Data extraction\nConfiguration: ")
    print(json.dumps(config, indent=2), end='\n')

    print("Reading index ...")
    index = pyndri.Index(config["indexed_data"])
    _, id2token, _ = index.get_dictionary()
    externalDocId = {index.document(doc)[0]: doc for doc in range(index.document_base(), index.maximum_document())}

    stoplist = set(stopwords.words("english")) if config["stopwords"] else {}

    print("Extract queries ...")
    queries = extractTopics(config["queries"])
    print("{n} queries to process.".format(n=len(queries)))

    queries_text = {}
    q_times = defaultdict(int)
    print("Preprocess queries ...")
    for q in tqdm(queries):
        q_text = clean(queries[q], config["stemmer"], stoplist)
        q_times[q_text] += 1  # queries with duplicate content
        queries_text[q] = q_text if q_times[q_text] == 1 else ' '.join([q_text, str(q_times[q_text])])
        if q_times[q_text] > 1:
            logging.warning("Duplicated: " + q + '\t' + ' '.join([q_text, str(q_times[q_text])]))

    out_trec_f = join(config["output_folder"], "trec_corpus.txt")
    out_t = codecs.open(out_trec_f, "w", encoding='utf8')

    print("Collection2Text ...")
    nl = 0
    relations = []

    qrels = get_qrels(config["relevance_judgements"])  # qrels[(q, doc)] = rel q:str, rel:int
    logging.info('From relevance judgements : ' + config["relevance_judgements"])
    k = config["title_len"]

    # read document titles:
    print("Writing trec_corpus ...")
    for e in tqdm(qrels):
        q = e[0]
        doc = e[1]
        rel = qrels[e]
        doc_txt = ' '.join([id2token[x] for x in index.document(externalDocId[doc])[1][:k] if x > 0])
        doc_txt = "xx" if doc_txt.strip() == '' else doc_txt
        out_t.write(q + ' ' + queries_text[q] + '\n')
        out_t.write(doc + ' ' + doc_txt + '\n')

    # corpus_txt of every fold:
    print("Writing corpus folds ...")
    for fold in tqdm(os.listdir(config["split_folds"])):
        for file in os.listdir(os.path.join(config["split_folds"], fold)):
            phase = file.split("_")[0]
            if not os.path.exists(os.path.join(config["output_folder"], fold)):
                os.mkdir(os.path.join(config["output_folder"], fold))
            phase_queries = [l.strip() for l in open(os.path.join(os.path.join(config["split_folds"],
                                                                               fold), file), 'r').readlines()]
            relations = [((e[0], e[1]), qrels[e]) for e in qrels if e[0] in phase_queries]
            with open(os.path.join(os.path.join(config["output_folder"], fold), "corpus_"+phase+".txt"), 'w') as out:
                for e in relations:
                    q = e[0][0]
                    doc = e[0][1]
                    doc_txt = ' '.join([id2token[x] for x in index.document(externalDocId[doc])[1][:k] if x > 0])
                    doc_txt = "xx" if doc_txt.strip() == '' else doc_txt
                    rel = e[1]
                    out.write(str(rel) + '\t' + queries_text[q] + '\t' + doc_txt + '\n')
    print("Done.")


