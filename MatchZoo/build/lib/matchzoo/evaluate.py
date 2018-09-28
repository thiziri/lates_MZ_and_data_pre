import pandas as pd
import pytrec_eval
import json
import sys
import numpy as np


def get_metric_results(qrel, run):
    evaluator = pytrec_eval.RelevanceEvaluator(qrel, {'map', 'ndcg_cut'})
    return evaluator.evaluate(run)

def mz_eval(mz_output_file):
    """Evaluates the metrics on a TREC format file output by MatchZoo

    parameters:
    ==========
    mz_output_file : string
        path to MatchZoo output TREC format file

    file_to_write : string
        where the results of the evaluation should be stored
        ignore if it is None
    """

    with open(mz_output_file) as f:
        df = pd.read_csv(f, sep='\t')

    Y_true = df.groupby(df.columns[[0]].tolist())[df.columns[
        [2, 6]].tolist()].apply(lambda g: dict(map(tuple, g.values.tolist()))).to_dict()
    Y_pred = df.groupby(df.columns[[0]].tolist())[df.columns[
        [2, 4]].tolist()].apply(lambda g: dict(map(tuple, g.values.tolist()))).to_dict()

    results = get_metric_results(Y_true, Y_pred)

    # print(json.dumps(results, indent=1))
    return results


if __name__ == '__main__':
    evaluation = mz_eval(sys.argv[1])  # give MZ.test.predict.txt file
    measures = evaluation[list(evaluation.keys())[0]].keys()
    average_evaluation_all = {measure:np.average([evaluation[q][measure] for q in evaluation]) for measure in measures}
    print(json.dumps(average_evaluation_all, indent=2))
