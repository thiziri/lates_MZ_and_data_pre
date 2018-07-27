import json
import random
import pyndri
import sys
import os

sys.path.append('../for_matchZoo/utils')

from tools4text import chunkIt, extractTopics, extract_trec_million_queries
from tqdm import tqdm
from os.path import join


if __name__ == '__main__':
    print("Split data into folds for training,tast and validation")
    config_file = sys.argv[1]
    config_all = json.load(open(config_file))
    config = config_all["parameters"]
    print('Config: '+json.dumps(config, indent=2))

    if config["only_queries"] or config["only_docs"]:
        to_split = []

        if config["only_queries"]:
            queries = extractTopics(config["queries_folder"]) if config["queries_format"] == "trec" \
                else extract_trec_million_queries(config["queries_folder"])
            to_split = list(queries.keys())
        elif config["only_docs"]:
            index = pyndri.Index(config["index"])
            docs_list = [doc for doc in range(index.document_base(), index.maximum_document())]
            external_doc_id = [index.document(doc)[0] for doc in docs_list]
            to_split = external_doc_id

        folds = {}
        random.shuffle(to_split)
        split = chunkIt(to_split, config["folds_num"])

        if config["validation"]:
            for i in tqdm(range(config["folds_num"])):
                    # print("fold ",i, end="\t")
                    test = split[i]
                    valid = split[0]
                    try:
                        valid = split[i + 1]
                    except:
                        pass
                    # print(test, valid)
                    train = list(set(to_split) - set(test).union(valid))
                    folds[i] = {"test": test, "valid": valid, "train": train}
        else:
            for i in tqdm(range(config["folds_num"])):
                # print("fold ",i, end="\t")
                test = split[i]
                train = list(set(to_split) - set(test))
                folds[i] = {"test": test, "train": train}
        if config["same_test_train_valid"]:
            folds[0] = {"test": to_split, "valid": to_split, "train": to_split}
        print("Saving folds ...")
        for i in tqdm(folds):
                f = join(config["output_folder"], "fold_" + str(i))
                os.mkdir(f)
                for group in folds[i]:
                    with open(join(f, group+"_.txt"), "w") as q_out:
                        q_out.write("\n".join([str(q) for q in folds[i][group]]))

        print("Saved data ok.")

    else:
        print("Configuration error.")
