import logging
from tqdm import tqdm
from collections import defaultdict
import json
import os
from os.path import join
import codecs
import sys
import pyndri
from nltk.corpus import stopwords
from tools4text import get_all_phrases, extract_trec_million_queries
from tools4text import clean, get_qrels_1, get_text_of_a_passage, read_values

logging.basicConfig(filename='logs/build_corpus_separate.log', level=logging.DEBUG)

if __name__ == '__main__':
    config_file = sys.argv[1]
    config = json.load(open(config_file))
    logging.info('Config: '+json.dumps(config, indent=2))

    print("Data extraction\nConfiguration: ")
    print(json.dumps(config, indent=2), end='\n')

    print("Reading document index ...")
    index = pyndri.Index(config["indexed_data"])
    _, id2token, _ = index.get_dictionary()
    externalDocId = {}
    for doc in range(index.document_base(), index.maximum_document()):
        extD, _ = index.document(doc)
        externalDocId[extD] = doc

    print("Extract queries ...")
    queries = extract_trec_million_queries(config["queries"])

    stoplist = set(stopwords.words("english")) if config["stopwords"] else {}
    qrels_MQ = []  # get only judged queries
    for file in os.listdir(config["qrels_MQ"]):
        qrels_MQ += list(read_values(os.path.join(config["qrels_MQ"], file), 0))
    qrels_MQ = set(qrels_MQ)

    print("Pre-process queries %d queries..." % len(qrels_MQ))
    q_times = defaultdict(int)
    logging.info("Pre-process queries %d queries..." % len(qrels_MQ))
    queries_text = {}
    for q in tqdm(qrels_MQ):
        q_text = clean(queries[q], config["stemmer"], stoplist)
        q_times[q_text] += 1  # queries with duplicate content
        queries_text[q] = q_text if q_times[q_text] == 1 else ' '.join([q_text, str(q_times[q_text])])
        if q_times[q_text] > 1:
            logging.warning("Duplicated: " + q + '\t' + ' '.join([q_text, str(q_times[q_text])]))

    out_trec_f = join(config["output_folder"], "trec_corpus.txt")
    out_t = codecs.open(out_trec_f, "w", encoding='utf8')
    qrels = {}
    for file in os.listdir(config["qrels_MQ"]):
        qrels.update(get_qrels_1(os.path.join(config["qrels_MQ"], file)))  # get all qrels files
    print("Qrels : ", list(qrels.keys())[0:10])
    logging.info("Total pairs: " + str(len(qrels)))

    print("Extraction of contextual content ...")
    for fold in tqdm(config["folds"]):  # fold0 ... fold5
        save = join(config["output_folder"], fold)
        if not os.path.exists(save):
            os.mkdir(save)
        for phase in config["folds"][fold]:  # train, test, valid
            corpus_file = open(join(save, "corpus_"+phase+".txt"), 'w')
            phase_queries = []  # queries list
            for set_ in config["folds"][fold][phase]:
                phase_queries += list(read_values(os.path.join(config["qrels_MQ"], set_ + "_qrels.txt"), 0))
            for q_id in phase_queries:
                out_t.write(q_id + ' ' + queries_text[q_id] + '\n')  # write the trec corpus
                passages = open(join(config["retrieved_passages"], q_id)).readlines()[:config["top_k"]]  # top passages
                # get passages text:
                unique_documents = defaultdict(list)
                last_doc_id = ''
                doc_txt = ''
                for line in passages:  # get text passages
                    doc_id = line.strip().split()[1]
                    int_id = externalDocId[doc_id]  # get internal id
                    begin_id = int(line.strip().split()[2])
                    end_id = int(line.strip().split()[3])
                    passage_txt = get_text_of_a_passage(int_id, index, id2token, [begin_id, end_id])
                    unique_documents[doc_id].append(passage_txt)  # to concat passages, or start new one

                # save results of the current q_id:
                i = 0
                for res in unique_documents:
                    i += 1
                    num = 0
                    for passage_txt in unique_documents[res]:
                        num += 1
                        # because we can have different passages from same document to different queries, or same
                        # passages from different documents to different queries
                        new_doc_id = "_".join([res, q_id, str(num)])
                        out_t.write(new_doc_id + ' ' + passage_txt + ' ' + q_id + '_' + str(num) + str(i) + '\n')
                        try:
                            rel = qrels[(q_id, res)]  # get relevance of the document
                        except:
                            rel = 0
                            logging.warning(str((q_id, res)) + " not found !")
                        corpus_file.write(str(rel)+'\t'+queries_text[q_id]+'\t'+passage_txt + ' ' + q_id + '_' +
                                          str(num) + '_' + str(i) + '\n')
                # res passages
            # phase_queries: train, test or valid queries
        # fold
    # all folds
    print("Finished.")
