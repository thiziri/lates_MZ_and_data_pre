import pyndri
from os.path import join
import os.path
import os, codecs
import docopt
from tools4text import extractTopics, clean, get_qrels, save_corpus, get_docs_from_run, run2relations
from tqdm import tqdm
import json
import sys
import random
from collections import defaultdict

import logging
logging.basicConfig(filename='collect2MZinpuText.log',level=logging.DEBUG)

# devide list seq into num different sub-lists
def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0
    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg
    return out

if __name__ == '__main__':
	config_file = sys.argv[1]
	#print(sys.argv[1])
	config = json.load(open(config_file))
	logging.info('Config: '+json.dumps(config, indent=2))


	print("Data extraction\nConfiguration: ")
	print(json.dumps(config, indent=2), end='\n')

	print("Reading index ...")
	index = pyndri.Index(config["indexed_data"])
	_, id2token, _ = index.get_dictionary()
	externelDocId = {}
	for doc in range(index.document_base(), index.maximum_document()):
		extD, _=index.document(doc)
		externelDocId[extD] = doc
	queries = extractTopics(config["queries"])
	queries_text = {}
	q_times = defaultdict(int)
	for q in queries:
		q_text = clean(queries[q], "krovetz",{})
		q_times[q_text]+=1
		queries_text[q] = q_text if q_times[q_text]==1 else ' '.join([q_text, str(q_times[q_text])])

	out_trec_f = join(config["output_folder"], "trec_corpus.txt")
	out_t = codecs.open(out_trec_f,"w",encoding='utf8')
	# qrels = get_qrels(config["relevance_judgements"]) # dictionary: "qrels[(q,doc)]:rel" with q and rel are ints 


	print("Collection2Text ...")
	nl = 0
	relations = []
	if config["from_qrels"]:#bool(config["relevance_judgements"]) and not bool(config["run_file"]) and not bool(config["runs_folder"]):
		qrels = get_qrels(config["relevance_judgements"])
		ranked_documents = set([e[1] for e in qrels])
		print("totalling: %d documents"% len(ranked_documents))
		nl = save_corpus(queries_text, ranked_documents, index, id2token, externelDocId, out_t)
		logging.info("Corpus file saved to " + out_trec_f+" with "+str(nl)+" lines")

		relations = [(e, qrels[e]) for e in qrels] # same content
		logging.info('From relevance judgements : ' + config["relevance_judgements"])

	elif config["from_run"]:#bool(config["run_file"]) and not bool(config["runs_folder"]):
		logging.info("From run: " + config["run_file"])
		ranked_documents = get_docs_from_run(config["run_file"])
		print("totalling: %d documents"% len(ranked_documents))
		nl = save_corpus(queries_text, ranked_documents, index, id2token, externelDocId, out_t)
		logging.info("Corpus file saved to " + out_trec_f+" with "+str(nl)+" lines")
		with open(config["run_file"],"r") as rank:
			queries_group = []
			ran = 0
			if bool(config["relevance_judgements"]):
				qrels = get_qrels(config["relevance_judgements"])
			for line in tqdm(rank):
				if line != None:
					# print(line)
					q = int(line.strip().split()[0])
					if q in queries_group:
						ran +=1
					else:
						ran = 1
						queries_group.append(q)
					doc = line.strip().split()[2]
					rel = 0
					if bool(config["relevance_judgements"]):
						try:
							rel = qrels[(q,doc)]
						except:
							logging.warning("No relevance info for ({q},{doc}), default 0".format(q=q, doc=doc))
					elif not config["binary_judgements"]:
						rel = 3 if ran in range(11) else 2 if ran in range(11, 31) else 1 if ran in range(31, 51) else 0
					else:
						rel = 1 if ran in range(8) else 0
					relations.append(((q,doc), rel))


	elif config["from_runs"]:#bool(config["runs_folder"]) :
		logging.info("From a set of runs in " + config["runs_folder"])
		ranked_documents = set()
		for f in os.listdir(config["runs_folder"]):
			ranked_documents = ranked_documents.union(get_docs_from_run(join(config["runs_folder"],f)))
		print("totalling: %d documents"% len(ranked_documents))
		nl = save_corpus(queries_text, ranked_documents, index, id2token, externelDocId, out_t)
		logging.info("Corpus file saved to " + out_trec_f+" with "+str(nl)+" lines")
		for f in os.listdir(config["runs_folder"]):
			with open(join(config["runs_folder"],f),"r") as rank:
				queries_group = []
				ran = 0
				for line in tqdm(rank):
					if line != None:
						# print(line)
						q = int(line.strip().split()[0])
						if q in queries_group:
							ran +=1
						else:
							ran = 1
							queries_group.append(q)
						doc = line.strip().split()[2]
						#rel = qrels[(q,doc)] # line.strip().split()[4]
						rel = 3 if ran in range(11) else 2 if ran in range(11, 31) else 1 if ran in range(31, 51) else 0
						relations.append(((q,doc), rel))
	# write corpus content
	if not config["cross_validation"]:
		nl = 0
		logging.info("Without cross-validation, resulting files are in " + config["output_folder"])
		out_f = join(config["output_folder"], "sample.txt")
		out = codecs.open(out_f,"w",encoding='utf8')
		for e in relations:
			q = str(e[0][0])
			doc = e[0][1]
			rel = e[1]
			try:
				doc_text = " ".join([id2token[x] for x in index.document(externelDocId[doc])[1] if x!=0])
			except:
				doc_text = ""
			if doc_text != "":
				out.write("{r}\t{q}\t{d}\n".format(r=rel, q=queries_text[q], d=doc_text, encoding='utf8'))
				nl += 1
		print("Collection2Text finished.\nResults in {f}\n{n} lines.".format(f=out_f, n=nl))

	else:
		# perform n_cross validation
		logging.info("Data for %d-cross-validation, resulting files are in %s" .format(config["num_folds"], config["output_folder"]))

		def select_rel_by_qids(qid_list, relations_list):
			# select relations 
			rel = [r for r in relations_list if r[0][0] in qid_list]
			return set(rel)

		n_folds = int(config["num_folds"])
		qid_group = list(set([r[0][0] for r in relations]))
		random.shuffle(qid_group)
		qid_group_folds = chunkIt(qid_group, n_folds)
		folds = {}
		for i in range(n_folds):
			# print("fold ",i, end="\t")
			qid_test = qid_group_folds[i]
			try:
				qid_valid = qid_group_folds[i+1]
			except:
				qid_valid = qid_group_folds[0]
			
			qid_train = list(set(qid_group) - set(qid_test).union(qid_valid))

			rel_train = select_rel_by_qids(qid_train, relations)
			rel_valid = select_rel_by_qids(qid_valid, relations)
			rel_test = set()
			if config["from_run"] or not config["reranking"]:
				rel_test = select_rel_by_qids(qid_test, relations)
			else:
				qrels = []
				if bool(config["relevance_judgements"]):
					qrels = get_qrels(config["relevance_judgements"])
				relations_test = run2relations(config["run_file"], config["binary_judgements"], qrels)
				rel_test = select_rel_by_qids(qid_test, relations_test)
			folds[i] = {"test":rel_test, "valid":rel_valid, "train":rel_train}
			# print("train: %s\t valid: %s\t test: %s"% (str(set([r[0][0] for r in rel_train])), str(set([r[0][0] for r in rel_valid])), str(set([r[0][0] for r in rel_test]))))

		# save relation files in different folds
		for i in tqdm(folds):
			f = join(config["output_folder"],"fold_"+str(i))
			os.mkdir(f)
			#print(folds[i])
			for group in folds[i]:
				out = open(join(f, "corpus_"+group+".txt"), "w")
				for r in folds[i][group]:
					q = str(r[0][0])
					doc = r[0][1]
					rel = r[1]
					try:
						doc_text = " ".join([id2token[x] for x in index.document(externelDocId[doc])[1] if x!=0])
					except:
						doc_text = ""
					if doc_text != "":
						out.write("{r}\t{q}\t{d}\n".format(r=rel, q=queries_text[q], d=doc_text, encoding='utf8'))
				out.close()