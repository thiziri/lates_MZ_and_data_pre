import pyndri
from os.path import join
import os
import codecs
import sys
import json
import logging
from tqdm import tqdm
from collections import defaultdict

sys.path.append('../utils')

from tools4text import extractTopics, clean, get_qrels, save_corpus, get_docs_from_run, run2relations
from tools4text import rank_to_relevance, path_leaf, remove_extension, extract_trec_million_queries


logging.basicConfig(filename='collect2MZinpuText.log',level=logging.DEBUG)

if __name__ == '__main__':
    config_file = sys.argv[1]
    config = json.load(open(config_file))
    logging.info('Config: '+json.dumps(config, indent=2))

    print("Data extraction\nConfiguration: ")
    print(json.dumps(config, indent=2), end='\n')

    print("Reading index ...")
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
    print("Preprocess queries ...")
    for q in tqdm(queries):
        q_text = clean(queries[q], "krovetz", {})
        q_times[q_text] += 1
        queries_text[q] = q_text if q_times[q_text] == 1 else ' '.join([q_text, str(q_times[q_text])])

    out_trec_f = join(config["output_folder"], "trec_corpus.txt")
    out_t = codecs.open(out_trec_f, "w", encoding='utf8')

    print("Collection2Text ...")
    nl = 0
    relations = []

    if config["from_qrels"]:
        qrels = get_qrels(config["relevance_judgements"])  # qrels[(q, doc)] = rel q:str, rel:int
        ranked_documents = set([e[1] for e in qrels])
        if bool(config["rerank_run"]):
            ranked_documents = ranked_documents.union(get_docs_from_run(config["rerank_run"]))
        print("totalling: %d documents" % len(ranked_documents))
        nl = save_corpus(queries_text, ranked_documents, index, id2token, externalDocId, out_t)

        logging.info("Corpus file saved to " + out_trec_f+" with "+str(nl)+" lines")

        relations = [((e[0], e[1]), qrels[e]) for e in qrels]  # same content (q, doc):rel, q:int
        logging.info('From relevance judgements : ' + config["relevance_judgements"])

    elif config["from_run"]:
        logging.info("From run: " + config["train_run"])
        qrels = get_qrels(config["relevance_judgements"]) if bool(config["relevance_judgements"]) else []

        ranked_documents = get_docs_from_run(config["train_run"])
        if bool(config["relevance_judgements"]):
            ranked_documents = ranked_documents.union(set([e[1] for e in qrels]))
        if bool(config["rerank_run"]):
            ranked_documents = ranked_documents.union(get_docs_from_run(config["rerank_run"]))
            
        print("totalling: %d documents" % len(ranked_documents))
        nl = save_corpus(queries_text, ranked_documents, index, id2token, externalDocId, out_t)
        logging.info("Corpus file saved to " + out_trec_f+" with "+str(nl)+" lines")
        if bool(config["train_relations"]):
            print("get train_relations ...")
            qrels_like = get_qrels(config["train_relations"])
            relations = [((e[0], e[1]), qrels_like[e]) for e in tqdm(qrels_like)]
        else:
            relations = run2relations(config["train_run"],
                                      config["binary_judgements"],
                                      qrels,
                                      config["scales"],
                                      config["ranks"],
                                      config["train_relations_k"])

    else:
        sys.exit("Configuration error: extract data from_run or from_qrels only.")

    # write corpus content
    if not config["cross_validation"]:
        nl = 0
        logging.info("Without cross-validation, resulting files are in " + config["output_folder"])
        out_f = join(config["output_folder"], "sample.txt")
        out = codecs.open(out_f, "w", encoding='utf8')
        for e in relations:
            q = e[0][0]
            doc = e[0][1]
            rel = e[1]
            try:
                doc_text = " ".join([id2token[x] for x in index.document(externalDocId[doc])[1] if x != 0])
            except:
                doc_text = ""
            if doc_text != "":
                out.write("{r}\t{q}\t{d}\n".format(r=rel, q=queries_text[q], d=doc_text, encoding='utf8'))
                nl += 1
        print("Collection2Text finished.\nResults in {f}\n{n} lines.".format(f=out_f, n=nl))

    else:
        # perform n_cross validation
        logging.info("Data cross-validation, resulting files are in %s" .format(config["output_folder"]))

        def select_rel_by_qids(qid_list, relations_list):
            # select relations 
            return set([re for re in tqdm(relations_list) if re[0][0] in qid_list])


        folds = {}
        qrels = get_qrels(config["relevance_judgements"]) if bool(config["relevance_judgements"]) else []
        # qrels[(str,str)]:int
        relations_test = []
        if config["reranking"]:
            print("Getting reranking relations ...")
            relations_test = run2relations(config["rerank_run"],
                                       config["binary_judgements"],
                                       qrels,
                                       config["scales"],
                                       config["ranks"],
                                       config["test_relations_k"])

        for fold in os.listdir(config["split_data"]):
            # print("fold ", fold)
            if os.path.isdir(join(config["split_data"], fold)):
                qid_test = [l.strip() for l in open(join(join(config["split_data"], fold), "test_.txt"),
                                                    'r').readlines()]
                qid_valid = [l.strip() for l in open(join(join(config["split_data"], fold), "valid_.txt"),
                                                     'r').readlines()]
                qid_train = [l.strip() for l in open(join(join(config["split_data"], fold), "train_.txt"),
                                                     'r').readlines()]

                print("Select relations train/test/valid by q_id ...")
                rel_train = select_rel_by_qids(qid_train, relations)
                rel_valid = select_rel_by_qids(qid_valid, relations)  # validation in part of the training relation set
                rel_test = set()
                if not config["reranking"]:
                    rel_test = select_rel_by_qids(qid_test, relations)
                else:
                    rel_test = select_rel_by_qids(qid_test, relations_test)
                folds[path_leaf(fold)] = {"test": rel_test, "valid": rel_valid, "train": rel_train}

        del relations
        del relations_test

        tr_vl_q_id = "mq" if config["train_queries_format"] == "mq" else ""
        tst_q_id = "mq" if config["test_queries_format"] == "mq" else ""

        print("save relation files in different folds ...")
        for fold in folds:
            f = join(config["output_folder"], fold)
            os.mkdir(f)
            for group in folds[fold]:
                out = open(join(f, "corpus_"+group+".txt"), "w")
                for r in tqdm(folds[fold][group]):
                    q = tr_vl_q_id + r[0][0] if group in {"valid", "train"} else tst_q_id + r[0][0]
                    doc = r[0][1]
                    rel = r[1]
                    doc_text = ""
                    try:
                        doc_text = " ".join([id2token[x] for x in index.document(externalDocId[doc])[1] if x != 0])
                    finally:
                        pass

                    if doc_text != "":
                        out.write("{r}\t{q}\t{d}\n".format(r=rel, q=queries_text[q], d=doc_text, encoding='utf8'))
                out.close()
