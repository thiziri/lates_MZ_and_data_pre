import pyndri
from os.path import join
import os.path
import os, codecs
import itertools
import docopt
from tools4text import extractTopics, clean, get_qrels, get_docs_from_run, save_corpus, rank2relations, pad
from tqdm import tqdm
import collections
from random import shuffle

"""
Split a sequence seq to size folds
"""
def split_seq(seq, size):
        newseq = []
        splitsize = 1.0/size*len(seq)
        for i in range(size):
                newseq.append(seq[int(round(i*splitsize)):int(round((i+1)*splitsize))])
        return newseq

if __name__ == '__main__':
	args = docopt.docopt("""
		Usage:
		    collection2corpus_relations_firstN.py --i=<indexed_data> --d=<data_name> --q=<queries> [--r=<relevance_judgements>] [--rank=<ranking_file>] [--ranklist=<runs_folder>] [--train_all --f=<folds_number>] [--pad] [--bin] --o=<output_folder> 

		Example:
		    collection2corpus_relations_firstN.py --i=/home/thiziri/Documents/DOCTORAT/COLLECTION/Indri_index/AP88  --d=AP88 --q=/home/thiziri/Documents/DOCTORAT/COLLECTION/TOPICS/AP88 --r=/home/thiziri/Documents/DOCTORAT/COLLECTION/QRELS/extractedQrels/qrels251_300AP88 --o='/home/thiziri/Documents/DOCTORAT/SOFT/MatchZoo/my_tests/custom_test/data/AP88/from_qrels'

		Options:
		    --i=<indexed_data>	Gives the INDRI index of the collection.
		    --d=<data_name	Give the data set name.
		    --q=<queries>	Gives the queries file.
		    --r=<relevance_judgements>	Gives the TREC like relevance judgements file.
		    --rank=<ranking_file>	Gives the TREC formatted ranking file.
		    --bin	if binary judgements, otherwise multiscale relevance is used.
		    --ranklist=<runs_folder>	Gives the folder that contains a set of TREC like ranking files.
		    --train_all	Add this if want to train queries in all judged documents. 
		    --f=<folds_number>	Will create f folds for cross validation based on all judged documents for training then re-rank documents.
		    --pad	If you want to padd training set with random documents in order to reach same size with predict set. 
		    --o=<output_folder>	Gives the output folder where constructed file will be stored.

		""")

	print("Data extraction")

	print("Reading index ...")
	index = pyndri.Index(args["--i"])
	_, id2token, _ = index.get_dictionary()
	externelDocId = {}
	for inD in range(index.document_base(), index.maximum_document()):
		extD, _=index.document(inD)
		externelDocId[extD] = inD
	queries = extractTopics(args["--q"])
	queries_text = {}
	for q in queries:
		queries_text[q] = clean(queries[q], "krovetz",{})

	out_f = join(args["--o"], "corpus.txt")
	out_r = join(args["--o"], "relation.txt")
	out_q = join(args["--o"], args["--d"]+"qrels.txt")
	out = codecs.open(out_f,"w",encoding='utf8')
	out2 = codecs.open(out_r,"w",encoding='utf8')
	out3 = codecs.open(out_q,"w",encoding='utf8')

	nl = 0
	nl2 = 0
	
	# construct relation.txt from input data:
		
	if bool(args["--r"]) and not bool(args["--rank"]) and not bool(args["--ranklist"]):
		print("From Qrels file: "+args["--r"])
		qrels = get_qrels(args["--r"])
		ranked_documents = set([e[1] for e in qrels])
		nl = save_corpus(queries_text, ranked_documents, index, id2token, externelDocId, out)
		"""
		print("Append qrels ...")
		for q,d in tqdm(itertools.product(list(queries_text.keys()),list(externelDocId.keys()))):
			if (q,d) not in qrels:
				qrels[(q,d)] = '0'
		"""
		print("Construct relation.txt ...")
		for c in tqdm(collections.OrderedDict(sorted(qrels.items()))):
			#out3.write("{q} 0 {d} {r}\n".format(q=c[0], d=c[1], r=qrels[c]))
			out2.write("{r} {q} {d}\n".format(r=qrels[c], q=c[0], d=c[1]))
			nl2 +=1

	elif bool(args["--rank"]) and not bool(args["--r"]):
		print("From rank file: "+args["--rank"])
		print("Construct relation.txt ...")
		ranked_documents = get_docs_from_run(args["--rank"])
		nl = save_corpus(queries_text, ranked_documents, index, id2token, externelDocId, out)
		relations = rank2relations(args["--rank"], bool(args["--bin"]), out3)
		for r in relations:
			rel,q,doc = r
			out2.write("{r} {q} {d}\n".format(r=rel, q=q, d=doc))
			nl2 += 1

	elif bool(args["--rank"]) and bool(args["--r"]) and not(bool(args["--train_all"])):
		print("From rank file {s} with relevance judgements in {t}".format(s=args["--rank"], t=args["--r"]))
		qrels = get_qrels(args["--r"])
		#print(qrels)
		ranked_documents = get_docs_from_run(args["--rank"])
		nl = save_corpus(queries_text, ranked_documents, index, id2token, externelDocId, out)
		print("Construct relation.txt ...")
		with open(args["--rank"],"r") as rank:
			i = 0
			for line in tqdm(rank):
				if line != None:
					i += 1
					q = int(line.strip().split()[0])
					doc = line.strip().split()[2]
					rel = qrels[(q,doc)] if (q,doc) in qrels else 0
					out3.write("{q} 0 {d} {r}\n".format(q=q, d=doc, r=rel))
					out2.write("{r} {q} {d}\n".format(r=rel, q=q, d=doc))
					nl2 += 1

	elif bool(args["--rank"]) and bool(args["--r"]) and bool(args["--train_all"]):
		print("From rank file {s} with relevance judgements in {t}\nQueries will be trained in all documents, then perform re-ranking".format(s=args["--rank"], t=args["--r"]))
		qrels = get_qrels(args["--r"])
		print(len(qrels))
		Q = list(set([e[0] for e in qrels]))
		shuffle(Q)
		folds = split_seq(Q, int(args["--f"]))
		qrels_1 = [e for e in qrels if qrels[e]==1]
		print(len(qrels_1))
		qrels_0 = list(set(qrels.keys()) - set(qrels_1))
		print(len(qrels_0))
		qrels_equi = {e:1 for e in qrels_1}
		for q in Q: 
			num_pos = len([e[0] for e in qrels_1 if e[0]==q]) # add same number of negative examples for each query
			#num_pos = num_pos*2
			for e in qrels_0:
				if e[0]==q:
					if num_pos>0:
						qrels_equi[e] = 0
						num_pos -=1
					else:
						break
		
		qrels_equi2 = qrels_equi
		if bool(args["--pad"]):
			qrels_equi2 = pad(qrels_equi, args["--rank"])

		#print(qrels)
		ranked_documents = set([e[1] for e in qrels]).union(get_docs_from_run(args["--rank"]))
		nl = save_corpus(queries_text, ranked_documents, index, id2token, externelDocId, out)
		print("Construct relation files ...")

		# train on all judged documents and for relation_test.txt perform re-ranking

		for idx,f in tqdm(enumerate(folds)):
			test = f
			valid = folds[idx+1] if idx<len(folds)-1 else folds[0]
			train = list(itertools.chain.from_iterable([e for e in folds if e not in [test, valid]]))
			fold = join(args["--o"], "fold_"+str(idx))
			if not os.path.exists(fold):
				os.makedirs(fold)
			tr = join(fold, "relation_train.txt")
			tst = join(fold, "relation_test.txt")
			vld = join(fold, "relation_valid.txt")
			prdct = join(fold, "relation_predict.txt")
			# write relation files
			with open(tr, "w") as out:
				#print("Train in: ", train)
				for q in tqdm(train):
					for d in qrels_equi2:
						if d[0]==q:
							line = "{rel} {q} {d}\n".format(rel=qrels_equi[d], q=q, d=d[1])
							out.write(line)
			with open(prdct, "w") as out: # predictions for top 1000
				#print("Test in: ", test)
				for q in tqdm(test):
					rank = open(args["--rank"],"r")
					for l in rank:
						if str(q)+" Q0" in l:
							line = "{rel} {q} {d}\n".format(rel=qrels[(q,l.strip().split()[2])] if (q,l.strip().split()[2]) in qrels else 0 , q=q, d=l.strip().split()[2])
							out.write(line)
			with open(vld, "w") as out:
				#print("Validate in: ", valid)
				for q in tqdm(valid):
					for d in qrels_equi: # test with top 1000
						if d[0]==q:
							line = "{rel} {q} {d}\n".format(rel=qrels_equi[d], q=q, d=d[1])
							out.write(line)
			with open(tst, "w") as out:
				#print("Validate in: ", valid)
				for q in tqdm(test):
					for d in qrels_equi: 
						if d[0]==q:
							line = "{rel} {q} {d}\n".format(rel=qrels_equi[d], q=q, d=d[1])
							out.write(line)


	elif bool(args["--ranklist"]) and not bool(args["--r"]):
		print("Combining different ranks ...")
		ranked_documents = set()
		relations = set()
		for f in os.listdir(args["--ranklist"]):
			ranked_documents = ranked_documents.union(get_docs_from_run(join(args["--ranklist"],f)))
			for r in tqdm(rank2relations(join(args["--ranklist"], f), bool(args["--bin"]), out3)):
				relations.add(r)
		#print(ranked_documents)
		nl = save_corpus(queries_text, ranked_documents, index, id2token, externelDocId, out)
		print("Construct relation.txt ...")
		for rel,q,doc in tqdm(relations):
			out2.write("{r} {q} {d}\n".format(r=rel, q=q, d=doc))
			nl2 += 1
		

	out.close()
	out2.close()
	out3.close()

	print("Collection2Text finished.\nResults in {f}\n{n} lines in corpus.txt\n{n2} lines in relation.txt\nDone.".format(f=args["--o"], n=nl,n2=nl2))