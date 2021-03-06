import logging
from tqdm import tqdm
from collections import defaultdict
import json
import os
from os.path import join
import codecs
import sys
import pyndri
sys.path.append('../utils')
from tools4text import get_all_phrases, extract_trec_million_queries, extractTopics
from tools4text import clean, get_qrels, run2relations


logging.basicConfig(filename='collection2contextual.log', level=logging.DEBUG)

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
    queries = {}
    if config["train_queries"] == config["test_queries"]:
        queries = extractTopics(config["train_queries"]) if config["train_queries_format"] == "trec"\
            else extract_trec_million_queries(config["train_queries"])
    else:
        train_queries = extractTopics(config["train_queries"]) if config["train_queries_format"] == "trec" \
            else extract_trec_million_queries(config["train_queries"])
        test_queries = extractTopics(config["test_queries"]) if config["test_queries_format"] == "trec" \
            else extract_trec_million_queries(config["test_queries"])
        queries = {**train_queries, **test_queries}
    print("{n} queries to process.".format(n=len(queries)))

    queries_text = {}
    q_times = defaultdict(int)
    print("Pre-process queries ...")
    for q in tqdm(queries):
        q_text = clean(queries[q], "krovetz", {})
        q_times[q_text] += 1
        queries_text[q] = q_text if q_times[q_text] == 1 else ' '.join([q_text, str(q_times[q_text])])

    out_trec_f = join(config["output_folder"], "trec_corpus.txt")
    out_t = codecs.open(out_trec_f, "w", encoding='utf8')
    qrels = get_qrels(config["relevance_judgements"])
    print("Qrels : ", list(qrels.keys())[0:10])

    print("Extraction of contextual content ...")
    for fold in tqdm(os.listdir(config["split_query_folders"])):  # fold_0 ... fold_5
        save = join(config["output_folder"], fold)
        os.mkdir(save)
        for phase in os.listdir(join(config["split_query_folders"], fold)):  # train, test, valdi files
            corpus_file = open(join(save, "corpus_"+phase.split(".")[0].replace('_', '')+".txt"), 'w')
            phase_queries = open(join(config["split_query_folders"], join(fold, phase))).read().split('\n')  # queries list
            for q_id in phase_queries:
                out_t.write(q_id+' '+queries_text[q_id]+'\n')  # write the trec corpus
                passages = open(join(config["retrieved_passages"], q_id)).readlines()
                unique_documents = []
                for line in passages:
                    d_id = line.strip().split()[1]
                    unique_documents.append(d_id)  # insert the document id to count its passages
                    begin_id = int(line.strip().split()[2])
                    end_id = int(line.strip().split()[3])
                    doc = [x for x in index.document(externalDocId[d_id])[1]]
                    passage_txt = " ".join([id2token[x] for x in doc[begin_id:end_id] if x != 0])
                    passage_id = unique_documents.count(d_id)
                    try:
                        rel = qrels[(q_id, d_id)]
                    except:
                        rel = 0
                    doc_id = d_id + "_p" + str(passage_id)
                    out_t.write(doc_id+" "+passage_txt + '\n')  # write the trec corpus
                    corpus_file.write(str(rel)+'\t'+queries_text[q_id]+'\t'+passage_txt+'\n')  # write the train corpus
                # passages
            # phase_queries
        # fold
    # all folds
    print("Finished.")
