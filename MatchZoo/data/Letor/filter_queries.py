# select queries with at least one relevant document
# input: relevance judgments, queries sets (S1, S2, S3, S4, S5)
# output: filtered sets of queries
import os
import json
import sys
from tqdm import tqdm
from tools4text import get_qrels


if __name__ == "__main__":
    config = json.load(open(sys.argv[1], 'r'))
    """
    relevance=<relevance_judgments> sets=<query_sets> --n=<files_name> --o=<output_folder>
        
        Options:
            relevance=<relevance_judgments>    Give the relevance judgments files .
            sets=<query_sets>    Give the queries sets folder .
            --n=<files_name_start>    String with whom all names of the different sets start with .
            --o=<output_folder>    Where results should be stored .
            
        """

    print(json.dumps(config, indent=2))

    # get relevance judgments
    print("Relevance judgments ...")
    judgments = {}
    labels = set()
    if os.path.isfile(config["relevance"]):
        judgments, labels = get_qrels(config["relevance"])
    elif os.path.isdir(config["relevance"]):
        for file in os.listdir(config["relevance"]):
            judgment_file, new_labels = get_qrels(os.path.join(config["relevance"], file))
            judgments.update(judgment_file)
            labels = labels | new_labels
    print("Judged: ", len(judgments), list(judgments.keys())[:10], labels)
    # print(judgments)

    # get queries sets:
    print("Queries sets ...")
    sets = {}
    for file in os.listdir(config["sets"]):
        if os.path.isfile(os.path.join(config["sets"], file)):
            if file.startswith(config["name"]):
                print(file)
                q_ids = set()
                with open(os.path.join(config["sets"], file), 'r') as set_f:
                    for line in tqdm(set_f):
                        q_ids.add(line.split()[1].split(':')[-1])
                sets[file.split('.')[0]] = q_ids
    print("Processed sets: ", list(sets.keys()))

    # Filtering:
    print("Filtering ...")
    filtered_sets = {}
    sf = 0
    s = 0
    for set_ in sets:
        print(set_)
        fq_ids = [q_id for q_id in tqdm(sets[set_]) if config["ids"]+q_id in judgments and
                  (len(set([judgments[config["ids"]+q_id][doc] for doc in judgments[config["ids"]+q_id]]) & labels) > 1
                   or 0 not in set([judgments[config["ids"]+q_id][doc] for doc in judgments[config["ids"]+q_id]]))
                  ]
        print(len(sets[set_]), len(fq_ids))
        sf += len(fq_ids)
        s += len(sets[set_])
        filtered_sets[set_] = fq_ids
    print(s, sf)

    # Save filtered sets:
    print("Saving ...")
    if not os.path.exists(config["output"]):
        os.mkdir(config["output"])
    for set_ in tqdm(filtered_sets):
<<<<<<< HEAD
        with open(os.path.join(config["output"], set_+"_qrels"+".txt"), 'w') as out:
            for q_id in filtered_sets[set_]:
                lines = '\n'.join([q_id + '\t0\t' + d_id + '\t' + str(judgments[q_id][d_id]) for d_id in judgments[q_id]])
=======
        with open(os.path.join(config["output"], set_+"_qrels_"+".txt"), 'w') as out:
            for q_id in filtered_sets[set_]:
                lines = '\n'.join([q_id + '\t0' + d_id + '\t' + str(judgments[q_id][d_id]) for d_id in judgments[q_id]])
>>>>>>> 994bd944e350343b5f056039e7c2e20270fd589f
                out.write(lines + '\n')


