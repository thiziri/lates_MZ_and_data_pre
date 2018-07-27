from os.path import join
import docopt
from tqdm import tqdm
from tools4text import get_qrels

def read_relations(file):
    relations = {}
    with open(file, "r") as f:
        for l in tqdm(f):
            rel = l.strip().split()[0]
            q = l.strip().split()[1]
            d = l.strip().split()[2]
            if q in relations:
                relations[q][d] = rel
            else:
                relations[q] = {}
                relations[q][d] = rel
    return (relations)

if __name__ == '__main__':
    args = docopt.docopt("""
        Usage:
            relation2qrels.py --r=<relation_file> --q=<qrels_file> --o=<output_folder>

        Example:
            relation2qrels.py --r=<relation_file> --q=<qrels_file> 

        Options:
            --r=<relation_file>    Relation file of MatchZoo.
            --q=<qrels_file>    Trec like qrels file.
            --o=<output_folder>    Where constructed file whil be stored.

        """)

    print("Qrels extraction ...")
    qrels = get_qrels(args["--q"])
    relations = read_relations(args["--r"])
    qrels_relations = set()
    out = join(args["--o"], "qrels.mz")
    with open(out, 'w') as f:
        for q in tqdm(relations):
            for d in relations[q]: 
                try:
                    r = qrels[(int(q),d)]
                except:
                    r = 0
                qrels_relations.add((q,d,r))
        for r in qrels_relations:
            f.write("{q} 0 {d} {r}\n".format(q=r[0], d=r[1], r=r[2]))
    print("Finished.")