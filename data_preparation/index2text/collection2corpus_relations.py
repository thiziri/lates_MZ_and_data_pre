import pyndri
from os.path import join
import os.path
import os, codecs
import itertools
import docopt
from tools4text import extractTopics, clean, get_qrels
from tqdm import tqdm
import collections

if __name__ == '__main__':
	args = docopt.docopt("""
		Usage:
		    collection2corpus_relations.py --i=<indexed_data> --d=<data_name> --q=<queries> [--r=<relevance_judgements> | --rank=<ranking_file> | --ranklist=<runs_folder>] --o=<output_folder> 

		Example:
		    collection2corpus_relations.py --i=/home/thiziri/Documents/DOCTORAT/COLLECTION/Indri_index/AP88  --d=AP88 --q=/home/thiziri/Documents/DOCTORAT/COLLECTION/TOPICS/AP88 --r=/home/thiziri/Documents/DOCTORAT/COLLECTION/QRELS/extractedQrels/qrels251_300AP88 --o='/home/thiziri/Documents/DOCTORAT/SOFT/MatchZoo/my_tests/custom_test/data/AP88/from_qrels'

		Options:
		    --i=<indexed_data>	Gives the INDRI index of the collection.
		    --d=<data_name	Give the data set name.
		    --q=<queries>	Gives the queries file.
		    --r=<relevance_judgements>	Gives the TREC like relevance judgements file.
		    --rank=<ranking_file>	Gives the TREC formatted ranking file.
		    --ranklist=<runs_folder>	Gives the folder that contains a set of TREC like ranking files. 
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

	print("Collection2Text ...")
	nl = 0
	nl2 = 0
	print("Construct corpus.txt ...")
	for q in tqdm(collections.OrderedDict(sorted(queries_text.items()))):
		out.write("{d} {d_txt}\n".format(d=q, d_txt=queries_text[q], encoding='utf8'))
		nl += 1
	for doc in tqdm(externelDocId):
		try:
			doc_text = " ".join([id2token[x] for x in index.document(externelDocId[doc])[1] if x!=0])
		except:
			doc_text = ""
		if doc_text == "":
			doc_text = "0"
		out.write("{d} {d_txt}\n".format(d=doc, d_txt=doc_text, encoding='utf8'))
		nl += 1
	if bool(args["--r"]):
		print("From Qrels file: "+args["--r"])
		qrels = get_qrels(args["--r"])
		print("Append qrels ...")
		for q,d in tqdm(itertools.product(list(queries_text.keys()),list(externelDocId.keys()))):
			if (q,d) not in qrels:
				qrels[(q,d)] = '0' 
		print("Construct relation.txt ...")
		for c in tqdm(collections.OrderedDict(sorted(qrels.items()))):
			out3.write("{q} 0 {d} {r}\n".format(q=c[0], d=c[1], r=qrels[c]))
			out2.write("{r} {q} {d}\n".format(r=qrels[c], q=c[0], d=c[1]))
			nl2 +=1

	elif bool(args["--rank"]):
		print("From rank file: "+args["--rank"])
		print("Construct relation.txt ...")
		with open(args["--rank"],"r") as rank:
			i = 0
			qdr = {}
			for line in tqdm(rank):
				if line != None:
					i += 1
					q = line.strip().split()[0]
					doc = line.strip().split()[2]
					#score = line.strip().split()[4]
					#rel = 3 if i<=10 else 2 if (i in range(11,21)) else 1 if (i in range(21,101)) else 0 # multiscale relevance
					rel = 1 if i<=10 else 0 # binary relevance
					out3.write("{q} 0 {d} {r}\n".format(q=q, d=doc, r=rel))
					out2.write("{r} {q} {d}\n".format(r=rel, q=q, d=doc))
					nl2 += 1

	elif bool(args["--ranklist"]):
		for f in os.listdir(args["--ranklist"]):
			with open(join(args["--ranklist"],f),"r") as rank:
				for line in tqdm(rank):
					if line != None:
						q = line.strip().split()[0]
						doc = line.strip().split()[2]
						rel = line.strip().split()[4]
						#?????????????????????

	out.close()

	print("Collection2Text finished.\nResults in {f}\n{n} lines in corpus.txt\n{n2} lines in relation.txt\nDone.".format(f=args["--o"], n=nl,n2=nl2))