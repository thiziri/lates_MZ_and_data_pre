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
from tools4text import get_all_phrases, extract_trec_million_queries, extractTopics
from tools4text import clean, get_qrels, get_text_of_a_passage, read_values

logging.basicConfig(filename='collection2concat_contexts.log', level=logging.DEBUG)

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

    q_times = defaultdict(int)
    print("Pre-process queries %d queries..." % len(qrels_MQ))
    logging.info("Pre-process queries %d queries..." % len(qrels_MQ))
    queries_text = {}
    for q in tqdm(qrels_MQ):
        q_text = clean(queries[q], config["stemmer"], stoplist)
        q_times[q_text] += 1  # queries with duplicate content
        queries_text[q] = q_text if q_times[q_text] == 1 else ' '.join([q_text, str(q_times[q_text])])
        if q_times[q_text] > 1:
            logging.info("Duplicated: " + q + '\t' + ' '.join([q_text, str(q_times[q_text])]))

    # ################################ heeeeeeeeeeeeeeeeeeeeere ########################

    out_trec_f = join(config["output_folder"], "trec_corpus.txt")
    out_t = codecs.open(out_trec_f, "w", encoding='utf8')
    qrels = get_qrels(config["relevance_judgements"])
    print("Qrels : ", list(qrels.keys())[0:10])

    print("Extraction of contextual content ...")
    for fold in tqdm(os.listdir(config["split_query_folders"])):  # fold_0 ... fold_5
        save = join(config["output_folder"], fold)
        if not os.path.exists(save):
            os.mkdir(save)
        for phase in os.listdir(join(config["split_query_folders"], fold)):  # train, test, valdi
            corpus_file = open(join(save, "corpus_"+phase.split(".")[0].replace('_', '')+".txt"), 'w')
            phase_queries = open(join(config["split_query_folders"], join(fold, phase))).read().split('\n')  # queries list
            for q_id in phase_queries:
                out_t.write(q_id+' '+queries_text[q_id]+'\n')  # write the trec corpus
                passages = open(join(config["retrieved_passages"], q_id)).readlines()[:config["top_k"]]  # top k passages
                # get passages text:
                unique_documents = {}
                last_doc_id = ''
                doc_txt = ''
                for line in passages:
                    doc_id = line.strip().split()[1]
                    int_id = externalDocId[doc_id]  # get internal id
                    begin_id = int(line.strip().split()[2])
                    end_id = int(line.strip().split()[3])
                    passage_txt = get_text_of_a_passage(int_id, index, id2token, [begin_id, end_id])
                    if doc_id in unique_documents:
                        unique_documents[doc_id].append(passage_txt)  # to concat passages
                    else:
                        unique_documents[doc_id] = [passage_txt]  # start new document
                # save results
                num = 0
                for res in unique_documents:
                    num += 1
                    new_doc_id = res + "_" + q_id  # because we can have diff passages from same doc to diff queries
                    doc_txt = " ".join(unique_documents[res])
                    out_t.write(new_doc_id + ' ' + doc_txt + ' ' + q_id + '_' + str(num) + '\n')  # write corpus the concatenated passages
                    try:
                        rel = qrels[(q_id, res)]  # get relevance of the document
                    except:
                        rel = 0
                    corpus_file.write(str(rel)+'\t'+queries_text[q_id]+'\t'+doc_txt + ' ' + q_id + '_' + str(num)+'\n')  # write the train corpus
                # concatenated passages
            # phase_queries: train, test or valid queries
        # fold
    # all folds
    print("Finished.")
